//! `rembg-rs` command line: `i` (one file), `p` (a folder), `d` (download models).

use std::fs;
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use rembg_rs::{download_models, model_names, remove_bytes, RemoveOptions, Session};

#[derive(Parser)]
#[command(name = "rembg-rs", version, about = "Remove image backgrounds")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// for a file as input
    I(FileArgs),
    /// for a folder as input
    P(DirArgs),
    /// download models
    D {
        /// models to download; every model when omitted
        models: Vec<String>,
    },
}

#[derive(clap::Args)]
struct Common {
    /// model name
    #[arg(short = 'm', long, default_value = "bria-rmbg", value_parser = clap::builder::PossibleValuesParser::new(model_names()))]
    model: String,
    /// use alpha matting
    #[arg(short = 'a', long)]
    alpha_matting: bool,
    /// trimap fg threshold
    #[arg(long, default_value_t = 240)]
    alpha_matting_foreground_threshold: u8,
    /// trimap bg threshold
    #[arg(long, default_value_t = 10)]
    alpha_matting_background_threshold: u8,
    /// erode size
    #[arg(long, default_value_t = 10)]
    alpha_matting_erode_size: usize,
    /// output only the mask
    #[arg(long)]
    only_mask: bool,
    /// post process the mask
    #[arg(long)]
    post_process_mask: bool,
    /// remove the background color fringing left on soft edges
    #[arg(long)]
    decontaminate: bool,
    /// background color (R G B A) to replace the removed background with
    #[arg(long = "bgcolor", value_names = ["R", "G", "B", "A"], num_args = 4)]
    bgcolor: Option<Vec<u8>>,
}

impl Common {
    fn options(&self) -> RemoveOptions {
        let bgcolor = match self.bgcolor.as_deref() {
            Some(c) => c.try_into().ok(),
            // python-rembg's CLI defaults --bgcolor to transparent black.
            None => Some([0, 0, 0, 0]),
        };
        RemoveOptions {
            alpha_matting: self.alpha_matting,
            alpha_matting_foreground_threshold: self.alpha_matting_foreground_threshold,
            alpha_matting_background_threshold: self.alpha_matting_background_threshold,
            alpha_matting_erode_size: self.alpha_matting_erode_size,
            only_mask: self.only_mask,
            post_process_mask: self.post_process_mask,
            decontaminate: self.decontaminate,
            bgcolor,
        }
    }
}

#[derive(clap::Args)]
struct FileArgs {
    #[command(flatten)]
    common: Common,
    /// input image, or `-` for stdin
    input: Option<PathBuf>,
    /// output image, or `-` for stdout
    output: Option<PathBuf>,
}

#[derive(clap::Args)]
struct DirArgs {
    #[command(flatten)]
    common: Common,
    /// folder of input images
    input: PathBuf,
    /// folder for the resulting PNGs
    output: PathBuf,
    /// delete each input file after processing
    #[arg(short = 'd', long)]
    delete_input: bool,
    #[arg(long, hide = true)]
    accept_usage_cost: bool,
    #[arg(short = 'w', long, hide = true)]
    watch: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::I(args) => run_file(args),
        Command::P(args) => run_dir(args),
        Command::D { models } => {
            let names: Vec<&str> = models.iter().map(String::as_str).collect();
            download_models(&names)
        }
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}

fn is_stdio(path: Option<&Path>) -> bool {
    match path {
        None => true,
        Some(p) => p == Path::new("-"),
    }
}

fn run_file(args: FileArgs) -> rembg_rs::session::Result<()> {
    let opts = args.common.options();

    let (input_bytes, input_name) = if is_stdio(args.input.as_deref()) {
        let mut buf = Vec::new();
        std::io::stdin().read_to_end(&mut buf)?;
        (buf, None)
    } else {
        let path = args.input.as_deref().expect("checked above");
        (fs::read(path)?, Some(path.to_path_buf()))
    };

    let mut out: Box<dyn Write> = if is_stdio(args.output.as_deref()) {
        match (input_name, std::io::stdout().is_terminal()) {
            (Some(path), true) => Box::new(fs::File::create(path.with_extension("out.png"))?),
            _ => Box::new(std::io::stdout().lock()),
        }
    } else {
        Box::new(fs::File::create(args.output.as_deref().expect("checked above"))?)
    };

    let mut session = Session::new(&args.common.model)?;
    out.write_all(&remove_bytes(&input_bytes, &mut session, &opts)?)?;
    Ok(())
}

fn run_dir(args: DirArgs) -> rembg_rs::session::Result<()> {
    let opts = args.common.options();
    let mut session = Session::new(&args.common.model)?;
    let mut inputs = Vec::new();
    collect(&args.input, &mut inputs)?;

    for input in inputs {
        let Some(name) = input.file_name() else { continue };
        let output = args.output.join(name).with_extension("png");
        if output.exists() {
            continue;
        }
        let bytes = match fs::read(&input) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("{e}");
                continue;
            }
        };
        if image::guess_format(&bytes).is_err() {
            continue;
        }
        if let Err(e) = fs::create_dir_all(output.parent().unwrap_or(Path::new("."))) {
            eprintln!("{e}");
            continue;
        }

        match remove_bytes(&bytes, &mut session, &opts) {
            Ok(png) => {
                if let Err(e) = fs::write(&output, png) {
                    eprintln!("{e}");
                }
            }
            Err(e) => eprintln!("{e}"),
        }

        if args.delete_input {
            let _ = fs::remove_file(&input);
        }
    }

    Ok(())
}

/// Every file under `dir`, recursively, like pathlib's `glob("**/*")`.
fn collect(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect(&path, out)?;
        } else {
            out.push(path);
        }
    }
    Ok(())
}
