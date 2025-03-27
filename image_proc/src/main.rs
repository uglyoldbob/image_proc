use std::io::Read;

use eframe::CreationContext;
use egui_plot::{Line, Plot, PlotPoints};

struct MainData {
    scale: Vec<f64>,
    loading: Option<String>,
    img: Option<eframe::egui::load::SizedTexture>,
}

impl MainData {
    fn new(_cc: &CreationContext) -> Self {
        Self {
            scale: vec![0.0; 16],
            loading: None,
            img: None,
        }
    }
}

impl eframe::App for MainData {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            egui_extras::install_image_loaders(ctx);

            if ui.button("Open image").clicked() {
                let f = rfd::FileDialog::new()
                    .add_filter("Image", &["jpg", "png"])
                    .set_directory("./")
                    .pick_file();
                if let Some(f) = f {
                    self.loading
                        .replace(format!("file://{}", f.to_str().unwrap()));
                }
            }

            if let Some(f) = &self.loading {
                let img = ctx.try_load_texture(
                    f,
                    eframe::egui::TextureOptions::LINEAR,
                    eframe::egui::load::SizeHint::Scale(1.0.into()),
                );
                if let Ok(i) = img {
                    match i {
                        eframe::egui::load::TexturePoll::Pending { size: _ } => {}
                        eframe::egui::load::TexturePoll::Ready { texture } => {
                            self.img.replace(texture);
                        }
                    }
                }
            }

            if let Some(ib) = &self.img {
                ui.add(eframe::egui::Image::from_texture(*ib));
            }

            let less_points = &self.scale;
            let spoints = self
                .scale
                .iter()
                .enumerate()
                .map(|(i, y)| {
                    splines::Key::new(i as f64, y.to_owned(), splines::Interpolation::Cosine)
                })
                .collect();
            let spline = splines::Spline::from_vec(spoints);
            let mut points_out = [0.0; 340];
            let time_scale = (less_points.len() - 1) as f64 / (points_out.len() - 1) as f64;
            for (i, e) in points_out.iter_mut().enumerate() {
                let t = time_scale * i as f64;
                let a: f64 = spline.clamped_sample(t).unwrap();
                *e = a;
            }
            let a: PlotPoints = points_out
                .iter_mut()
                .enumerate()
                .map(|(i, v)| [time_scale * i as f64, *v])
                .collect();
            let plot: PlotPoints = less_points
                .iter()
                .enumerate()
                .map(|a| [a.0 as f64, *a.1])
                .collect();
            let line = Line::new(plot);
            let line2 = Line::new(a);
            let p = Plot::new("my_plot")
                .view_aspect(2.0)
                .allow_drag(false)
                .show(ui, |plot_ui| {
                    plot_ui.line(line);
                    plot_ui.line(line2);
                });
            if p.response.clicked() {
                if let Some(ptr) = p.response.interact_pointer_pos() {
                    let a = p.transform.value_from_position(ptr);
                    println!("Plot point is at {:?}", a);
                }
            } else if p.response.dragged() {
                if let Some(ptr) = p.response.interact_pointer_pos() {
                    let a = p.transform.value_from_position(ptr);
                    let b = a.x.round();
                    if b >= 0.0 && b < self.scale.len() as f64 {
                        let newpos = ptr + p.response.drag_delta();
                        let newpos2 = p.transform.value_from_position(newpos);
                        self.scale[b as usize] = newpos2.y as f64;
                    }
                }
            }
        });
    }
}

fn main() {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default(),
        ..Default::default()
    };
    eframe::run_native(
        "Uob Image Processing",
        options,
        Box::new(|cc| Ok(Box::new(MainData::new(cc)))),
    )
    .unwrap();
}
