#[cfg(feature = "zvec")]
fn main() {
    use std::env;
    use std::path::PathBuf;

    let header_path = env::var("ZVEC_HEADER")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("third_party/zvec/include/zvec.h"));

    if !header_path.exists() {
        println!(
            "cargo:warning=ZVEC feature enabled but header not found at {}",
            header_path.display()
        );
        return;
    }

    let bindings = bindgen::Builder::default()
        .header(header_path.to_string_lossy().to_string())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate ZVEC bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    bindings
        .write_to_file(out_path.join("zvec_bindings.rs"))
        .expect("Couldn't write ZVEC bindings");

    println!("cargo:rerun-if-env-changed=ZVEC_HEADER");
    println!("cargo:rerun-if-changed={}", header_path.display());

    if let Ok(lib_dir) = env::var("ZVEC_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", lib_dir);
    }

    if let Ok(lib_name) = env::var("ZVEC_LIB_NAME") {
        println!("cargo:rustc-link-lib={}", lib_name);
    } else {
        println!("cargo:rustc-link-lib=zvec");
    }
}

#[cfg(not(feature = "zvec"))]
fn main() {}
