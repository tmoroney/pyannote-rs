use pyannote_rs::{EmbeddingExtractor, EmbeddingManager};
use std::time::Instant;
use sysinfo::{System, RefreshKind, MemoryRefreshKind, CpuRefreshKind, Pid};

fn main() {
    let audio_path = std::env::args().nth(1).expect("Please specify audio file");
    
    // Initialize system monitoring
    let mut system = System::new_with_specifics(
        RefreshKind::new()
            .with_memory(MemoryRefreshKind::new())
            .with_cpu(CpuRefreshKind::new())
    );
    let pid = std::process::id() as usize;
    
    fn print_memory_usage(system: &System, pid: usize, label: &str) {
        if let Some(process) = system.process(Pid::from(pid)) {
            let memory_mb = process.memory() as f64 / 1024.0 / 1024.0;
            println!("💾 Memory usage {}: {:.2} MB", label, memory_mb);
        }
    }
    
    // Initial memory usage
    print_memory_usage(&system, pid, "initial");
    
    // Timer: Audio loading
    let start_time = Instant::now();
    let (samples, sample_rate) = pyannote_rs::read_wav(&audio_path).unwrap();
    system.refresh_process(Pid::from(pid));
    print_memory_usage(&system, pid, "after audio loading");
    println!("⏱️  Audio loading: {:.2}ms", start_time.elapsed().as_millis());
    
    let max_speakers = 6;

    // Timer: Model initialization
    let start_time = Instant::now();
    let mut extractor = EmbeddingExtractor::new("wespeaker_en_voxceleb_CAM++.onnx").unwrap();
    let mut manager = EmbeddingManager::new(6);
    system.refresh_process(Pid::from(pid));
    print_memory_usage(&system, pid, "after model initialization");
    println!("⏱️  Model initialization: {:.2}ms", start_time.elapsed().as_millis());

    // Timer: Segmentation
    let start_time = Instant::now();
    let segments: Vec<_> = pyannote_rs::get_segments(&samples, sample_rate, "segmentation-3.0.onnx")
        .unwrap()
        .collect();
    let segment_count = segments.len();
    system.refresh_process(Pid::from(pid));
    print_memory_usage(&system, pid, "after segmentation");
    println!("⏱️  Segmentation: {:.2}ms (found {} segments)", start_time.elapsed().as_millis(), segment_count);

    // Timer: Embedding computation and speaker identification
    let embedding_start = Instant::now();
    let mut embedding_time = 0u64;
    let mut speaker_id_time = 0u64;
    let mut processed_segments = 0usize;
    
    for segment in segments {
        match segment {
            Ok(segment) => {
                // Timer: Individual embedding computation
                let embed_start = Instant::now();
                if let Ok(embedding) = extractor.compute(&segment.samples) {
                    embedding_time += embed_start.elapsed().as_millis() as u64;
                    processed_segments += 1;
                    
                    // Timer: Speaker identification
                    let speaker_start = Instant::now();
                    let speaker = if manager.get_all_speakers().len() == max_speakers {
                        manager
                            .get_best_speaker_match(embedding.collect())
                            .map(|s| s.to_string())
                            .unwrap_or("?".into())
                    } else {
                        manager
                            .search_speaker(embedding.collect(), 0.5)
                            .map(|s| s.to_string())
                            .unwrap_or("?".into())
                    };
                    speaker_id_time += speaker_start.elapsed().as_millis() as u64;
                    
                    println!(
                        "start = {:.2}, end = {:.2}, speaker = {}",
                        segment.start, segment.end, speaker
                    );
                } else {
                    embedding_time += embed_start.elapsed().as_millis() as u64;
                    println!(
                        "start = {:.2}, end = {:.2}, speaker = ?",
                        segment.start, segment.end
                    );
                }
            }
            Err(error) => eprintln!("Failed to process segment: {:?}", error),
        }
    }
    
    println!("⏱️  Total embedding computation: {:.2}ms (avg {:.2}ms per segment)", 
             embedding_time, embedding_time as f64 / processed_segments as f64);
    println!("⏱️  Total speaker identification: {:.2}ms (avg {:.2}ms per segment)", 
             speaker_id_time, speaker_id_time as f64 / processed_segments as f64);
    println!("⏱️  Total processing (embedding + speaker ID): {:.2}ms", 
             embedding_start.elapsed().as_millis());
    
    // Final memory usage
    system.refresh_process(Pid::from(pid));
    print_memory_usage(&system, pid, "final");
}
