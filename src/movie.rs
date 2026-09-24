//! High-level access to a VQA movie: parse the container once, then iterate
//! decoded video frames or decode the soundtrack without hand-composing the
//! chunk parsers, the padding rule, the FINF transforms, or the per-version
//! stereo layouts.

use nom::Parser;
use nom::bytes::complete::tag;

use crate::audio::{CodecState, decompress_into};
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
                    let data = chunk.data;
                    if !stereo {
                        decompress_into(&mut left, data, &mut samples);
                    } else if self.header.version == VQAVersion::Three {
                        // v3 splits the chunk in halves: left then right
                        let (l, r) = data.split_at(data.len() / 2);
                        decode_stereo(&mut left, &mut right, l.iter().zip(r), &mut samples);
                        // an odd length leaves the right half a byte longer:
                        // decode it to keep that predictor in step, but its
                        // samples have no left partner and are dropped
                        if let Some(&unpaired) = r.get(l.len()) {
                            right.decode_byte(unpaired);
                        }
                    } else {
                        // v1/v2 alternate bytes (two nibble-samples each)
                        // between left and right
                        let (pairs, rest) = data.as_chunks::<2>();
                        let pairs = pairs.iter().map(|[l, r]| (l, r));
                        decode_stereo(&mut left, &mut right, pairs, &mut samples);
                        // likewise an odd length ends on an unpaired left byte
                        if let [unpaired] = rest {
                            left.decode_byte(*unpaired);
                        }
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

/// Decode stereo byte pairs straight into interleaved samples: each
/// (left, right) byte pair yields `l0 r0 l1 r1`. The channels' predictors
/// are independent, so decoding them side by side lets the CPU overlap the
/// two dependency chains.
fn decode_stereo<'a>(
    left: &mut CodecState,
    right: &mut CodecState,
    pairs: impl ExactSizeIterator<Item = (&'a u8, &'a u8)>,
    samples: &mut Vec<i16>,
) {
    samples.reserve(pairs.len() * 4);
    for (&l, &r) in pairs {
        let [l0, l1] = left.decode_byte(l);
        let [r0, r1] = right.decode_byte(r);
        samples.extend_from_slice(&[l0, r0, l1, r1]);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::decompress;

    /// A minimal movie: the header plus one SND2 chunk per entry of `sound`.
    fn movie(version: u16, channels: u8, sound: &[Vec<u8>]) -> Vec<u8> {
        let mut header = Vec::new();
        header.extend(version.to_le_bytes());
        header.extend(1u16.to_le_bytes()); // flags: has sound
        header.extend(0u16.to_le_bytes()); // frames
        header.extend(8u16.to_le_bytes()); // width
        header.extend(4u16.to_le_bytes()); // height
        header.extend([4, 2, 15, 0]); // block size, frame rate, cbparts
        header.extend(256u16.to_le_bytes()); // colors
        header.extend(0u16.to_le_bytes()); // maxblocks
        header.extend([0; 6]); // unk1, unk2
        header.extend(22050u16.to_le_bytes());
        header.extend([channels, 16]);
        header.extend([0; 14]); // unk3, unk4, max_cbfz_size, unk5
        assert_eq!(header.len(), 42);

        let mut body = b"WVQA".to_vec();
        for (id, data) in
            std::iter::once((b"VQHD", &header)).chain(sound.iter().map(|d| (b"SND2", d)))
        {
            body.extend(id);
            body.extend((data.len() as u32).to_be_bytes());
            body.extend(data);
            if data.len() % 2 == 1 {
                body.push(0);
            }
        }
        let mut file = b"FORM".to_vec();
        file.extend((body.len() as u32).to_be_bytes());
        file.extend(body);
        file
    }

    /// The stereo layouts as first written: split each chunk into one byte
    /// run per channel, decode the runs separately, and zip the samples.
    fn split_decode_zip(version: u16, channels: u8, sound: &[Vec<u8>]) -> Vec<i16> {
        let (mut left, mut right) = (CodecState::new(), CodecState::new());
        let mut samples = Vec::new();
        for data in sound {
            if channels < 2 {
                samples.extend(decompress(&mut left, data));
                continue;
            }
            let (l, r): (Vec<u8>, Vec<u8>) = if version == 3 {
                let (l, r) = data.split_at(data.len() / 2);
                (l.to_vec(), r.to_vec())
            } else {
                (
                    data.iter().step_by(2).copied().collect(),
                    data.iter().skip(1).step_by(2).copied().collect(),
                )
            };
            let l = decompress(&mut left, &l);
            let r = decompress(&mut right, &r);
            for (&l, &r) in l.iter().zip(&r) {
                samples.extend([l, r]);
            }
        }
        samples
    }

    #[test]
    fn decode_audio_matches_per_channel_decoding_in_every_layout() {
        // xorshift64 noise; odd lengths leave an unpaired byte whose
        // channel state must still advance for the following chunks
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let sound: Vec<Vec<u8>> = [7, 64, 1, 33, 0, 100, 5]
            .iter()
            .map(|&len| {
                (0..len)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        state as u8
                    })
                    .collect()
            })
            .collect();

        for version in [1, 2, 3] {
            for channels in [1, 2] {
                let file = movie(version, channels, &sound);
                let vqa = VQA::parse(&file).unwrap();
                assert_eq!(
                    vqa.decode_audio().unwrap(),
                    split_decode_zip(version, channels, &sound),
                    "version {version}, {channels} channel(s)"
                );
            }
        }
    }
}
