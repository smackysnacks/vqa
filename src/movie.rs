//! High-level access to a VQA movie: parse the container once, then iterate
//! decoded video frames or decode the soundtrack without hand-walking the
//! chunks, the FINF transforms, or the per-version stereo layouts.

use crate::audio::{CodecState, decompress_into, westwood};
use crate::chunk::{Chunk, Chunks};
use crate::error::Error;
use crate::header::{FrameInfo, VQAHeader, VQAVersion};
use crate::video::{Frame, FrameDecoder, FrameRef};

/// The most samples [`VQA::decode_audio`] collects.
const MAX_SOUNDTRACK_SAMPLES: usize = 1 << 26;

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
    /// where `body` starts in the file
    body_offset: usize,
    /// The CIND entries of the CINF chunk, the frames where each codebook
    /// takes over; see [`Frames`]
    codebook_schedule: &'a [u8],
}

impl<'a> VQA<'a> {
    /// Parse the container: FORM chunk, WVQA signature, and header. The rest
    /// of the movie is walked lazily by [`VQA::chunks`], [`VQA::frames`] and
    /// [`VQA::decode_audio`].
    pub fn parse(buffer: &'a [u8]) -> Result<VQA<'a>, Error> {
        // the FORM chunk's size is not trusted: v1 movies store less than
        // the file holds, so everything after the signature is walked
        let (form, rest) = buffer.split_first_chunk::<12>().ok_or(Error::Parse)?;
        let (form_id, form) = form.split_at(4);
        let (form_size, signature) = form.split_at(4);
        if form_id != b"FORM" || signature != b"WVQA" {
            return Err(Error::Parse);
        }
        let form_size = u32::from_be_bytes(form_size.try_into().expect("four bytes"));

        let mut chunks = Chunks::at(rest, 12);
        let vqhd = chunks.next().ok_or(Error::Parse)??;
        if &vqhd.id != b"VQHD" {
            return Err(Error::Parse);
        }
        let header = VQAHeader::parse(vqhd.data)?;
        let (body, body_offset) = chunks.remaining();

        // the frame index (FINF) and codebook schedule (CINF) sit between
        // the header and the first frame's data, possibly behind chunks we
        // have no use for (LINF, PINF, ...)
        let mut frame_index = None;
        let mut codebook_schedule: &[u8] = &[];
        for chunk in Chunks::at(body, body_offset).map_while(Result::ok) {
            match &chunk.id {
                b"CINF" => codebook_schedule = cind_entries(&chunk),
                b"FINF" => {
                    let (entries, _) = chunk.data.as_chunks::<4>();
                    frame_index = Some(
                        entries
                            .iter()
                            .map(|&entry| FrameInfo::from_raw(u32::from_le_bytes(entry)))
                            .collect(),
                    );
                    break;
                }
                _ => {}
            }
        }

        Ok(VQA {
            form_size,
            header,
            frame_index,
            body,
            body_offset,
            codebook_schedule,
        })
    }

    /// Iterate over every chunk following the header, in file order, with
    /// offsets counted from the start of the file.
    pub fn chunks(&self) -> Chunks<'a> {
        Chunks::at(self.body, self.body_offset)
    }

    /// Iterate over the movie's video frames, decoded in order.
    pub fn frames(&self) -> Result<Frames<'a>, Error> {
        Ok(Frames {
            chunks: self.chunks(),
            decoder: FrameDecoder::new(&self.header)?,
            // with no part count in the header, codebook parts complete
            // where the CINF schedule starts a codebook
            codebook_schedule: if self.header.cbparts == 0 {
                self.codebook_schedule
            } else {
                &[]
            },
            frame: 0,
            done: false,
        })
    }

    /// Decode the whole soundtrack into interleaved signed 16-bit samples
    /// ([`VQAHeader::num_channels`] channels at [`VQAHeader::sample_rate`]
    /// Hz): IMA ADPCM (`SND2`) in its per-version stereo layouts, Westwood
    /// ADPCM (`SND1`), and raw PCM (`SND0`). Fails on the first malformed
    /// chunk, where [`VQA::audio_chunks`] keeps the sound before it, and
    /// with [`Error::TooLarge`] past 2^26 samples (over 25 minutes of stereo
    /// sound at 22050 Hz): Westwood ADPCM can expand 64-fold, so a small
    /// crafted file could ask for gigabytes. `audio_chunks` holds only one
    /// chunk at a time, and has no such limit.
    pub fn decode_audio(&self) -> Result<Vec<i16>, Error> {
        self.decode_audio_up_to(MAX_SOUNDTRACK_SAMPLES)
    }

    fn decode_audio_up_to(&self, max_samples: usize) -> Result<Vec<i16>, Error> {
        let mut chunks = self.audio_chunks();
        let mut samples = Vec::new();
        while let Some(result) = chunks.next_into(&mut samples) {
            result?;
            if samples.len() > max_samples {
                return Err(Error::TooLarge("soundtrack"));
            }
        }
        Ok(samples)
    }

    /// The frames where a new codebook takes over, as the movie's CINF
    /// chunk schedules them (its CIND entries), in file order; none without
    /// one. [`Frames`] follows them for movies whose header gives no
    /// codebook part count (`cbparts` 0). Callers driving a
    /// [`FrameDecoder`] themselves call
    /// [`FrameDecoder::swap_in_codebook_parts`] before decoding each of
    /// these frames.
    pub fn codebook_starts(&self) -> impl Iterator<Item = u16> + 'a {
        self.codebook_schedule
            .as_chunks::<6>()
            .0
            .iter()
            .map(|entry| u16::from_le_bytes([entry[0], entry[1]]))
    }

    /// Decode the soundtrack one sound chunk at a time: what
    /// [`VQA::decode_audio`] returns, split into each `SND?` chunk's
    /// samples. It yields every chunk before a malformed one, so a damaged
    /// or cut-off movie still gives up the sound it has.
    pub fn audio_chunks(&self) -> AudioChunks<'a> {
        AudioChunks {
            chunks: self.chunks(),
            version: self.header.version,
            stereo: self.header.num_channels() >= 2,
            pcm16: self.header.bit_depth() == 16,
            left: CodecState::new(),
            right: CodecState::new(),
            bytes: Vec::new(),
            done: false,
        }
    }
}

/// Iterator over a movie's soundtrack, one sound chunk at a time: each item
/// holds one `SND?` chunk's samples, interleaved signed 16-bit as
/// [`VQA::decode_audio`] returns them. Stops after the first error.
///
/// Cloning saves the decoding position, IMA ADPCM predictors included.
#[derive(Debug, Clone)]
pub struct AudioChunks<'a> {
    chunks: Chunks<'a>,
    version: VQAVersion,
    stereo: bool,
    /// whether raw SND0 samples are 16-bit (else unsigned 8-bit)
    pcm16: bool,
    left: CodecState,
    right: CodecState,
    /// scratch space for unsigned 8-bit SND1 samples
    bytes: Vec<u8>,
    done: bool,
}

impl AudioChunks<'_> {
    /// Decode the next sound chunk and append its samples to `samples`:
    /// [`Iterator::next`] without a new buffer for every chunk.
    pub fn next_into(&mut self, samples: &mut Vec<i16>) -> Option<Result<(), Error>> {
        if self.done {
            return None;
        }
        loop {
            let chunk = match self.chunks.next()? {
                Ok(chunk) => chunk,
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            };
            match &chunk.id {
                b"SND2" => self.decode_snd2(chunk.data, samples),
                b"SND1" => {
                    self.bytes.clear();
                    westwood::decompress_into(chunk.data, &mut self.bytes);
                    samples.extend(self.bytes.iter().map(|&b| widen(b)));
                }
                // raw PCM: signed 16-bit, or unsigned 8-bit
                b"SND0" if self.pcm16 => samples.extend(
                    chunk
                        .data
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|&b| i16::from_le_bytes(b)),
                ),
                b"SND0" => samples.extend(chunk.data.iter().map(|&b| widen(b))),
                _ => continue,
            }
            return Some(Ok(()));
        }
    }

    fn decode_snd2(&mut self, data: &[u8], samples: &mut Vec<i16>) {
        let (left, right) = (&mut self.left, &mut self.right);
        if !self.stereo {
            decompress_into(left, data, samples);
        } else if self.version == VQAVersion::Three {
            // v3 splits the chunk in halves: left then right
            let (l, r) = data.split_at(data.len() / 2);
            decode_stereo(left, right, l.iter().zip(r), samples);
            // an odd length leaves the right half a byte longer: decode it
            // to keep that predictor in step, but its samples have no left
            // partner and are dropped
            if let Some(&unpaired) = r.get(l.len()) {
                right.decode_byte(unpaired);
            }
        } else {
            // v1/v2 alternate bytes (two nibble-samples each) between left
            // and right
            let (pairs, rest) = data.as_chunks::<2>();
            let pairs = pairs.iter().map(|[l, r]| (l, r));
            decode_stereo(left, right, pairs, samples);
            // likewise an odd length ends on an unpaired left byte
            if let [unpaired] = rest {
                left.decode_byte(*unpaired);
            }
        }
    }
}

impl Iterator for AudioChunks<'_> {
    type Item = Result<Vec<i16>, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut samples = Vec::new();
        Some(self.next_into(&mut samples)?.map(|()| samples))
    }
}

/// Widen an unsigned 8-bit sample (0x80 silence) to signed 16-bit.
fn widen(sample: u8) -> i16 {
    (i16::from(sample) - 128) << 8
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

/// Iterator over a movie's decoded video frames. Every VQFR chunk (or
/// VQFK, a key frame) yields one frame, as does every pointer table of the
/// older layout that has frame sub-chunks at the top level; VQFL codebook
/// refreshes and the CINF codebook schedule are applied transparently.
/// Stops after the first error.
///
/// Each item is a copy of the decoder's frame. [`Frames::next_ref`] borrows
/// it instead, and [`Iterator::nth`] skips frames without copying them.
/// Cloning saves the decoding position: frames build on the state their
/// predecessors left, so a clone is the way back to an earlier frame
/// without starting over.
#[derive(Clone)]
pub struct Frames<'a> {
    chunks: Chunks<'a>,
    decoder: FrameDecoder,
    /// the CIND entries not yet reached
    codebook_schedule: &'a [u8],
    /// the number of the next frame
    frame: usize,
    done: bool,
}

/// One frame's data: a VQFR (or VQFK) chunk of sub-chunks, or the pointer
/// table ending a frame of the older top-level layout.
enum FrameData<'a> {
    Container(Chunk<'a>),
    Table(Chunk<'a>),
}

impl<'a> Frames<'a> {
    /// Decode the next frame and borrow it from the decoder: [`Iterator::next`]
    /// without the copy. The frame is valid until the next call.
    pub fn next_ref(&mut self) -> Option<Result<FrameRef<'_>, Error>> {
        if self.done {
            return None;
        }
        // a codebook the CINF schedule starts at this frame takes over
        // before any of the frame's chunks are read, in either layout
        let data = match self.start_frame().map(|()| self.next_frame_data()) {
            Ok(None) => return None,
            Ok(Some(Ok(data))) => data,
            Ok(Some(Err(e))) | Err(e) => {
                self.done = true;
                return Some(Err(e));
            }
        };
        self.frame += 1;
        let result = match data {
            FrameData::Container(chunk) => self.decoder.decode_chunks(chunk.sub_chunks()),
            FrameData::Table(chunk) => self
                .decoder
                .frame_chunk(&chunk)
                .and_then(|()| self.decoder.end_frame()),
        };
        self.done = result.is_err();
        Some(result)
    }

    /// Walk the chunks up to the next frame, applying VQFL refreshes and
    /// top-level codebook and palette chunks on the way.
    fn next_frame_data(&mut self) -> Option<Result<FrameData<'a>, Error>> {
        loop {
            let chunk = match self.chunks.next()? {
                Ok(chunk) => chunk,
                Err(e) => return Some(Err(e)),
            };
            let applied = match &chunk.id {
                b"VQFR" | b"VQFK" => return Some(Ok(FrameData::Container(chunk))),
                b"VQFL" => self.decoder.apply_side_chunks(chunk.sub_chunks()),
                // the older layout, without VQFR containers: each frame's
                // sub-chunks at the top level, ending at its pointer table
                b"VPT0" | b"VPTZ" | b"VPTK" | b"VPTD" => {
                    return Some(Ok(FrameData::Table(chunk)));
                }
                b"CBF0" | b"CBFZ" | b"CBP0" | b"CBPZ" | b"CPL0" | b"CPLZ" => {
                    self.decoder.frame_chunk(&chunk)
                }
                _ => Ok(()),
            };
            if let Err(e) = applied {
                return Some(Err(e));
            }
        }
    }

    /// Swap in the codebook parts staged so far if the CINF schedule starts
    /// a codebook at the next frame.
    fn start_frame(&mut self) -> Result<(), Error> {
        // each entry: the little-endian start frame, then a compressed size
        while let Some((entry, rest)) = self.codebook_schedule.split_first_chunk::<6>() {
            let start = usize::from(u16::from_le_bytes([entry[0], entry[1]]));
            if start > self.frame {
                break;
            }
            self.codebook_schedule = rest;
            if start == self.frame {
                self.decoder.swap_in_codebook_parts()?;
            }
        }
        Ok(())
    }
}

/// The entries of a CINF chunk's nested CIND chunk, 6 bytes each (a
/// little-endian `u16` start frame and `u32` compressed codebook size), or
/// none if the chunk doesn't parse.
fn cind_entries<'a>(cinf: &Chunk<'a>) -> &'a [u8] {
    let cind = cinf
        .sub_chunks()
        .map_while(Result::ok)
        .find(|chunk| &chunk.id == b"CIND");
    cind.map_or(&[], |chunk| chunk.data.as_chunks::<6>().0.as_flattened())
}

impl Iterator for Frames<'_> {
    type Item = Result<Frame, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.next_ref()?.map(|frame| frame.to_frame()))
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        // skipped frames are decoded but never copied out; as with the
        // default nth, an error among them ends the iteration
        for _ in 0..n {
            if self.next_ref()?.is_err() {
                return None;
            }
        }
        self.next()
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
    fn decode_audio_fails_past_its_sample_cap() {
        // three mono chunks of 3 bytes, 6 samples each
        let file = movie(2, 1, &[vec![0x77; 3], vec![0x77; 3], vec![0x77; 3]]);
        let vqa = VQA::parse(&file).unwrap();
        assert_eq!(vqa.decode_audio_up_to(18).map(|s| s.len()), Ok(18));
        assert_eq!(
            vqa.decode_audio_up_to(17),
            Err(Error::TooLarge("soundtrack"))
        );
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
