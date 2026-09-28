#![no_main]

use libfuzzer_sys::fuzz_target;
use vqa::*;

/// Walk every chunk in `chunks` and, `depth` levels down, the chunks nested
/// in them, checking that each one's offset points at its header and
/// payload in `input`, the bytes the walk's offsets count from.
fn walk(input: &[u8], chunks: Chunks<'_>, depth: usize) {
    for chunk in chunks {
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(e) => {
                // an error points into the input, at a chunk ID if it has one
                let offset = e.offset().expect("a chunk walk error has an offset");
                assert!(offset < input.len());
                if let Some(id) = e.chunk() {
                    assert_eq!(&input[offset..][..4], &id);
                }
                return;
            }
        };
        assert_eq!(&input[chunk.offset..][..4], &chunk.id);
        assert!(chunk.id.iter().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit()));
        let size = u32::from_be_bytes(input[chunk.offset + 4..][..4].try_into().unwrap());
        assert_eq!(size as usize, chunk.data.len());
        assert_eq!(&input[chunk.offset + 8..][..chunk.data.len()], chunk.data);
        if depth > 0 {
            walk(input, chunk.sub_chunks(), depth - 1);
        }
    }
}

fuzz_target!(|data: &[u8]| {
    // The standalone parsers must fail cleanly on arbitrary input, and the
    // chunk walk must describe the input faithfully.
    let _ = VQAHeader::parse(data);
    if let Some(&[a, b]) = data.first_chunk() {
        let number = u16::from_le_bytes([a, b]);
        if let Ok(version) = VQAVersion::try_from(number) {
            assert_eq!(u16::from(version), number);
        }
    }
    for &entry in data.as_chunks::<4>().0 {
        let info = FrameInfo::from_raw(u32::from_le_bytes(entry));
        assert_eq!(info.offset % 2, 0);
    }
    walk(data, Chunks::new(data), 2);

    // The high-level API must also hold up: parse, decode a bounded number
    // of video frames and convert them to RGB, and decode the soundtrack.
    // Borrowed frames (next_ref), skipped ones (nth), and a cloned iterator
    // must all agree with the owned frames.
    if let Ok(vqa) = VQA::parse(data) {
        if let (Ok(owned), Ok(mut borrowed)) = (vqa.frames(), vqa.frames()) {
            let mut frames = Vec::new();
            for frame in owned.take(16) {
                let view = borrowed.next_ref().expect("borrowed frames ended early");
                assert_eq!(frame.as_ref().map(Frame::view), view.as_ref().copied());
                if let Ok(frame) = &frame {
                    let _ = frame.to_rgb888();
                }
                let failed = frame.is_err();
                frames.push(frame);
                if failed {
                    break;
                }
            }
            // `frames` holds the first 16 frames, or all of them if the
            // movie ended or failed sooner, so it knows every nth below 16
            for n in [0, 3, 15] {
                let mut skipping = vqa.frames().expect("frames() succeeded above");
                assert_eq!(skipping.nth(n).as_ref(), frames.get(n));
            }
            // an error names the frame it ended, and points into the file
            if let Some(Err(e)) = frames.last() {
                assert_eq!(e.frame(), Some(frames.len() - 1));
                if let Some(offset) = e.offset() {
                    assert!(offset < data.len());
                }
            }
            if frames.len() > 2 {
                let mut original = vqa.frames().expect("frames() succeeded above");
                original.nth(1);
                let mut clone = original.clone();
                assert_eq!(original.next(), clone.next());
            }
        }
        let _ = vqa.decode_audio();
    }

    // Walk the movie the way a real consumer does: its chunks and their
    // sub-chunks, with offsets counted from the start of the file, then the
    // chunk each decoded FINF offset points at.
    let Ok(vqa) = VQA::parse(data) else {
        return;
    };
    walk(data, vqa.chunks(), 2);
    for frame in vqa.frame_index.iter().flatten() {
        let Some(frame_data) = data.get(frame.offset as usize..) else {
            continue;
        };
        walk(frame_data, Chunks::new(frame_data), 1);
    }
});
