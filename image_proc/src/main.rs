use std::{f64::consts::E, io::Read};

use eframe::{CreationContext, egui::Color32};
use egui_plot::{Line, Plot, PlotPoints};

struct MainData {
    scale: Vec<f64>,
    actual_image: Option<eframe::egui::ColorImage>,
    img: Option<eframe::egui::TextureHandle>,
    corrected_img: Option<eframe::egui::TextureHandle>,
}

enum InterpolationMethod {
    Nearest,
    Linear,
}

impl MainData {
    fn new(_cc: &CreationContext) -> Self {
        Self {
            scale: vec![0.0; 32],
            actual_image: None,
            img: None,
            corrected_img: None,
        }
    }

    fn correct_image(
        &mut self,
        ctx: &eframe::egui::Context,
        spline: splines::Spline<f64, f64>,
        method: InterpolationMethod,
    ) {
        if let Some(i) = &self.actual_image {
            let mut ic = i.clone();
            let mut moved_pixels: Vec<(f64, f64, eframe::egui::Color32)> = Vec::new();
            let center_x = ic.width() as f64 / 2.0;
            let center_y = ic.height() as f64 / 2.0;
            let corner_distance = ((center_x * center_x) + (center_y * center_y)).sqrt();
            for x in 0..ic.width() {
                for y in 0..ic.height() {
                    let dx = (x as f64) - center_x;
                    let dy = (y as f64) - center_y;
                    let dist = ((dx * dx) + (dy * dy)).sqrt() / corner_distance;
                    let correction = 1.0 + spline.clamped_sample(dist).unwrap();
                    let newdx = dx * correction;
                    let newdy = dy * correction;
                    let newx = newdx + center_x;
                    let newy = newdy + center_y;
                    moved_pixels.push((newx, newy, ic.pixels[y * ic.width() + x]));
                }
            }
            let width = ic.width();
            for i in &mut ic.pixels {
                *i = Color32::BLACK;
            }
            match method {
                InterpolationMethod::Linear => todo!(),
                InterpolationMethod::Nearest => {
                    let mut pmap: Vec<Option<Color32>> = vec![None; ic.width() * ic.height()];
                    for (x, y, p) in moved_pixels {
                        let nx = x.round();
                        let ny = y.round();
                        if nx >= 0.0
                            && ny >= 0.0
                            && nx < ic.width() as f64
                            && ny < ic.height() as f64
                        {
                            let nx = nx as usize;
                            let ny = ny as usize;
                            pmap[ny * width + nx] = Some(p);
                        }
                    }
                    for (i, p) in pmap.iter().enumerate() {
                        if let Some(p) = *p {
                            ic.pixels[i] = p;
                        } else {
                            ic.pixels[i] = Color32::MAGENTA;
                        }
                    }
                }
            }
            let a = ctx.load_texture("corrected_image", ic, eframe::egui::TextureOptions::LINEAR);
            self.corrected_img.replace(a);
        }
    }
}

fn generate_aruco_image() {
    let dict = opencv::aruco::DICT_4X4_100;
    let d = opencv::aruco::Dictionary::get(dict);
    println!("Getting dictionary");
    if let Ok(d) = d {
        println!("Making aruco board");
        let board = opencv::aruco::CharucoBoard::create(10, 10, 10.0 * 0.0254, 7.0 * 0.0254, &d);
        if let Ok(mut board) = board {
            println!("Saving aruco board");
            let mut pic = opencv::core::Mat::default();
            let a = opencv::aruco::CharucoBoardTrait::draw(
                &mut board,
                opencv::core::Size {
                    width: 2400,
                    height: 2400,
                },
                &mut pic,
                10,
                1,
            );
            let b = opencv::imgcodecs::imwrite("./aruco.png", &pic, &opencv::core::Vector::new());
            println!("Results {:?} {:?}", a, b);
        } else {
            println!("Failed to make arudo board {:?}", board);
        }
    }
}

impl eframe::App for MainData {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            egui_extras::install_image_loaders(ctx);

            eframe::egui::ScrollArea::vertical().show(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Open image").clicked() {
                        let f = rfd::FileDialog::new()
                            .add_filter("Image", &["jpg", "png"])
                            .set_directory("./")
                            .pick_file();
                        if let Some(f) = &f {
                            let f2 = std::fs::File::open(f);
                            if let Ok(mut f) = f2 {
                                let mut c = Vec::new();
                                let _ = f.read_to_end(&mut c);
                                let img = egui_extras::image::load_image_bytes(&c);
                                if let Ok(img) = img {
                                    let a = ctx.load_texture(
                                        "actual_image",
                                        img.clone(),
                                        eframe::egui::TextureOptions::LINEAR,
                                    );
                                    self.actual_image.replace(img);
                                    self.img.replace(a);
                                }
                            }
                        }
                    }
                    if ui.button("Generate aruco pattern").clicked() {
                        generate_aruco_image();
                    }
                });
                let w = ui.available_width();
                ui.horizontal(|ui| {
                    if let Some(th) = &self.img {
                        let z = w / th.size_vec2().x;
                        let st = eframe::egui::load::SizedTexture {
                            id: th.id(),
                            size: th.size_vec2() * z * 0.5,
                        };
                        ui.add(eframe::egui::Image::from_texture(st));
                    }

                    if let Some(th) = &self.corrected_img {
                        let z = w / th.size_vec2().x;
                        let st = eframe::egui::load::SizedTexture {
                            id: th.id(),
                            size: th.size_vec2() * z * 0.5,
                        };
                        ui.add(eframe::egui::Image::from_texture(st));
                    }
                });

                let less_points = &self.scale;
                let s = (self.scale.len() - 1) as f64;
                let spoints = self
                    .scale
                    .iter()
                    .enumerate()
                    .map(|(i, y)| {
                        splines::Key::new(
                            i as f64 / s,
                            y.to_owned(),
                            splines::Interpolation::Cosine,
                        )
                    })
                    .collect();
                let spline = splines::Spline::from_vec(spoints);
                let mut points_out = [0.0; 340];
                let time_scale = 1.0 / (points_out.len() - 1) as f64;
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
                    .map(|a| [a.0 as f64 / s, *a.1])
                    .collect();
                let line = Line::new(plot);
                let line2 = Line::new(a);
                let p = Plot::new("my_plot")
                    .view_aspect(2.0)
                    .auto_bounds([true, true])
                    .allow_drag(false)
                    .allow_scroll(false)
                    .allow_boxed_zoom(false)
                    .center_y_axis(true)
                    .show(ui, |plot_ui| {
                        plot_ui.line(line);
                        plot_ui.line(line2);
                    });
                if p.response.clicked() {
                    if let Some(ptr) = p.response.interact_pointer_pos() {
                        let a = p.transform.value_from_position(ptr);
                        println!("Plot point is at {:?}", a);
                    }
                } else if p.response.dragged_by(eframe::egui::PointerButton::Primary) {
                    if let Some(ptr) = p.response.interact_pointer_pos() {
                        let a = p.transform.value_from_position(ptr);
                        let b = (a.x * s).round();
                        if b >= 0.0 && b < self.scale.len() as f64 {
                            let newpos = ptr + p.response.drag_delta();
                            let newpos2 = p.transform.value_from_position(newpos);
                            self.scale[b as usize] = newpos2.y as f64;
                            self.correct_image(ctx, spline, InterpolationMethod::Nearest);
                        }
                    }
                }
            });
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
