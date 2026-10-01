//! Integration test: parse the container structures of the bundled
//! wwlogo.vqa and verify them against the file's known layout.

use vqa::{Chunks, FrameInfo, VQA, VQAHeader, VQAVersion};

#[test]
fn parses_wwlogo_header_and_frame_index() {
    let buffer = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/wwlogo.vqa"))
        .expect("failed to read wwlogo.vqa");

    let vqa = VQA::parse(&buffer).expect("failed to parse the container");
    assert_eq!(vqa.form_size as usize, buffer.len() - 8);

    let header = &vqa.header;
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

    // the same header, walked by hand: FORM, its size and WVQA, then the
    // VQHD chunk
    assert_eq!(&buffer[..4], b"FORM");
    assert_eq!(&buffer[8..12], b"WVQA");
    let vqhd = Chunks::new(&buffer[12..]).next().unwrap().unwrap();
    assert_eq!(&vqhd.id, b"VQHD");
    assert_eq!(&VQAHeader::parse(vqhd.data).unwrap(), header);

    // HiColor-era chunks (LINF, CINF) sit between the header and FINF
    let chunks: Vec<_> = vqa.chunks().map(Result::unwrap).collect();
    let finf = chunks.iter().find(|chunk| &chunk.id == b"FINF").unwrap();
    let (entries, _) = finf.data.as_chunks::<4>();
    let index: Vec<FrameInfo> = entries
        .iter()
        .map(|&entry| FrameInfo::from_raw(u32::from_le_bytes(entry)))
        .collect();
    assert_eq!(index.len(), usize::from(header.num_frames));
    assert_eq!(vqa.frame_index.as_ref(), Some(&index));

    // Each frame's data starts with an SN2J sound chunk (or a VQFL
    // full-codebook chunk at scene cuts); every decoded offset must land
    // exactly on one, in increasing order
    let mut previous = 0;
    for frame in &index {
        let offset = frame.offset as usize;
        assert!(offset > previous, "frame offsets must increase");
        let chunk = chunks.iter().find(|chunk| chunk.offset == offset).unwrap();
        assert!(&chunk.id == b"SN2J" || &chunk.id == b"VQFL");
        assert!(!frame.has_palette, "HiColor movies carry no palettes");
        previous = offset;
    }
    assert_eq!(index[0].offset, 682);
}
