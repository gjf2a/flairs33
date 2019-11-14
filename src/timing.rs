#[macro_export]
macro_rules! timed_op {
    ($label:expr, $line:stmt) => {
        println!("Started {}...", $label);
        let start = Instant::now();
        $line
        println!("Finished {} after {} seconds", $label, Instant::now().duration_since(start).as_secs());
    }
}