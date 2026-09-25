//! Integration test: parse the container structures of the bundled
//! wwlogo.vqa and verify them against the file's known layout.

use nom::Parser;
use nom::bytes::complete::{tag, take_until};

use vqa::{VQAVersion, finf_chunk, form_chunk, vqa_header};

#[test]
fn parses_wwlogo_header_and_frame_index() {
    let buffer = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/wwlogo.vqa"))
        .expect("failed to read wwlogo.vqa");

    let (rest, form) = form_chunk(&buffer).expect("failed to parse FORM chunk");
    assert_eq!(form.size as usize, buffer.len() - 8);

    let (rest, _) = tag::<_, _, nom::error::Error<_>>("WVQA")
        .parse(rest)
        .expect("missing WVQA signature");

    let (rest, header) = vqa_header(rest).expect("failed to parse VQA header");
    assert!(matches!(header.version, VQAVersion::Three));
    assert_eq!(header.flags, 0x001d);
    assert!(header.has_sound());
    assert_eq!(header.num_frames, 130);
    assert_eq!((header.width, header.height), (640, 400));
    assert_eq!((header.block_width, header.block_height), (4, 2));
    assert_eq!(header.frame_rate, 15);
    assert_eq!(header.freq, 22050);
    assert_eq!(header.channels, 2);
    assert_eq!(header.bits, 16);

    // HiColor-era chunks (LINF, CINF) sit between the header and FINF
    let (rest, _) = take_until::<_, _, nom::error::Error<_>>("FINF")
        .parse(rest)
        .expect("no FINF chunk found");
    let (_, finf) = finf_chunk(rest).expect("failed to parse FINF chunk");

    assert_eq!(finf.frames.len(), usize::from(header.num_frames));

    // Each frame's data starts with an SN2J sound chunk (or a VQFL
    // full-codebook chunk at scene cuts); every decoded offset must land
    // exactly on one, in increasing order
    let mut previous = 0;
    for frame in &finf.frames {
        let offset = frame.offset as usize;
        assert!(offset > previous, "frame offsets must increase");
        let fourcc = &buffer[offset..offset + 4];
        assert!(fourcc == b"SN2J" || fourcc == b"VQFL");
        assert!(!frame.has_palette, "HiColor movies carry no palettes");
        previous = offset;
    }
    assert_eq!(finf.frames[0].offset, 682);
}
