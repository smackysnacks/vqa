//! High-level access to a VQA movie: parse the container once, then iterate
//! decoded video frames or decode the soundtrack without hand-composing the
//! chunk parsers, the padding rule, the FINF transforms, or the per-version
//! stereo layouts.

use nom::Parser;
use nom::bytes::complete::tag;

use crate::audio::{CodecState, decompress};
use crate::error::Error;
use crate::parser::{
    FrameInfo, RawChunk, VQAHeader, VQAVersion, form_chunk, frame_info, raw_chunk, vqa_header,
};
use crate::video::{Frame, FrameDecoder};

/// A parsed VQA movie, borrowing the file's bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VQA<'a> {
    /// The FORM container size
    pub form_size: u32,
    /// The parsed movie header
    pub header: VQAHeader,
    /// The decoded FINF frame index (absolute byte offsets of each frame's
    /// data), when the movie carries one
    pub frame_index: Option<Vec<FrameInfo>>,
    /// Every chunk after the header, walked by [`VQA::chunks`]
    body: &'a [u8],
}

impl<'a> VQA<'a> {
    /// Parse the container: FORM chunk, WVQA signature, and header. The rest
    /// of the movie is walked lazily by [`VQA::chunks`], [`VQA::frames`] and
    /// [`VQA::decode_audio`].
    pub fn parse(buffer: &'a [u8]) -> Result<VQA<'a>, Error> {
        let (rest, form) = form_chunk(buffer).map_err(|_| Error::Parse)?;
        let (rest, _) = tag::<_, _, nom::error::Error<&[u8]>>("WVQA")
            .parse(rest)
            .map_err(|_| Error::Parse)?;
        let (body, header) = vqa_header(rest).map_err(|_| Error::Parse)?;

        // the FINF chunk sits between the header and the first frame's data,
        // possibly behind chunks we have no parser for (LINF, CINF, ...)
        let frame_index = Chunks { input: body }
            .take_while(Result::is_ok)
            .flatten()
            .find(|chunk| &chunk.id == b"FINF")
            .map(|chunk| {
                nom::multi::count(frame_info, chunk.data.len() / 4)
                    .parse(chunk.data)
                    .map(|(_, frames)| frames)
                    .map_err(|_| Error::Parse)
            })
            .transpose()?;

        Ok(VQA {
            form_size: form.size,
            header,
            frame_index,
            body,
        })
    }

    /// Iterate over every chunk following the header, in file order.
    pub fn chunks(&self) -> Chunks<'a> {
        Chunks { input: self.body }
    }

    /// Iterate over the movie's video frames, decoded in order.
    pub fn frames(&self) -> Result<Frames<'a>, Error> {
        Ok(Frames {
            chunks: self.chunks(),
            decoder: FrameDecoder::new(&self.header)?,
            done: false,
        })
    }

    /// Decode the whole soundtrack into interleaved signed 16-bit samples
    /// ([`VQAHeader::num_channels`] channels at [`VQAHeader::sample_rate`]
    /// Hz), handling the per-version stereo layouts of SND2 data and raw
    /// SND0 PCM. SND1 (Westwood ADPCM) is not supported yet.
    pub fn decode_audio(&self) -> Result<Vec<i16>, Error> {
        let stereo = self.header.num_channels() >= 2;
        let mut left = CodecState::new();
        let mut right = CodecState::new();
        let mut samples = Vec::new();

        for chunk in self.chunks() {
            let chunk = chunk?;
            match &chunk.id {
                b"SND2" => {
                    if !stereo {
                        samples.extend(decompress(&mut left, chunk.data));
                    } else if self.header.version == VQAVersion::Three {
                        // v3 splits the chunk in halves: left then right
                        let half = chunk.data.len() / 2;
                        let l = decompress(&mut left, &chunk.data[..half]);
                        let r = decompress(&mut right, &chunk.data[half..]);
                        interleave(&mut samples, &l, &r);
                    } else {
                        // v1/v2 alternate bytes (two nibble-samples each)
                        // between left and right
                        let lb: Vec<u8> = chunk.data.iter().step_by(2).copied().collect();
                        let rb: Vec<u8> = chunk.data.iter().skip(1).step_by(2).copied().collect();
                        let l = decompress(&mut left, &lb);
                        let r = decompress(&mut right, &rb);
                        interleave(&mut samples, &l, &r);
                    }
                }
                b"SND0" => {
                    // raw PCM: signed 16-bit, or unsigned 8-bit widened
                    if self.header.bit_depth() == 16 {
                        samples.extend(
                            chunk
                                .data
                                .as_chunks::<2>()
                                .0
                                .iter()
                                .map(|&b| i16::from_le_bytes(b)),
                        );
                    } else {
                        samples.extend(chunk.data.iter().map(|&b| (i16::from(b) - 128) << 8));
                    }
                }
                b"SND1" => return Err(Error::UnsupportedSound("SND1 (Westwood ADPCM)")),
                _ => {}
            }
        }

        Ok(samples)
    }
}

fn interleave(samples: &mut Vec<i16>, left: &[i16], right: &[i16]) {
    for (&l, &r) in left.iter().zip(right) {
        samples.push(l);
        samples.push(r);
    }
}

/// Iterator over the chunks of a movie body. Yields an [`Error::Parse`] and
/// then stops if the stream desyncs.
#[derive(Debug, Clone)]
pub struct Chunks<'a> {
    input: &'a [u8],
}

impl<'a> Iterator for Chunks<'a> {
    type Item = Result<RawChunk<'a>, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.input.is_empty() {
            return None;
        }
        match raw_chunk(self.input) {
            Ok((rest, chunk)) => {
                self.input = rest;
                Some(Ok(chunk))
            }
            Err(_) => {
                self.input = &[];
                Some(Err(Error::Parse))
            }
        }
    }
}

/// Iterator over a movie's decoded video frames. Every VQFR chunk yields one
/// frame; VQFL codebook refreshes are applied transparently. Stops after the
/// first error.
pub struct Frames<'a> {
    chunks: Chunks<'a>,
    decoder: FrameDecoder,
    done: bool,
}

impl Iterator for Frames<'_> {
    type Item = Result<Frame, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        loop {
            match self.chunks.next()? {
                Ok(chunk) => match &chunk.id {
                    b"VQFL" => {
                        if let Err(e) = self.decoder.process_vqfl(chunk.data) {
                            self.done = true;
                            return Some(Err(e));
                        }
                    }
                    b"VQFR" => {
                        let result = self.decoder.decode_frame(chunk.data);
                        self.done = result.is_err();
                        return Some(result);
                    }
                    _ => {}
                },
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            }
        }
    }
}
