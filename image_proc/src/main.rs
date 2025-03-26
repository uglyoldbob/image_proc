use std::f64::consts::PI;

use eframe::CreationContext;
use egui_plot::{Line, Plot, PlotPoints};

mod widget1;

struct MainData {}

impl MainData {
    fn new(_cc:&CreationContext) -> Self {
        Self {

        }
    }
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    }
    else {
        (x * PI).sin() / (x * PI)
    }
}

fn sinc_interpolate(indata: &[f64], odata: &mut [f64]) {
    let max_out = odata.len() - 1;
    for (i, elem) in odata.iter_mut().enumerate() {
        let o_frac = i as f64 / max_out as f64;
        let i_index = o_frac * (indata.len() - 1) as f64;
        let mut total : f64 = 0.0;
        for (index, in_element) in indata.iter().enumerate() {
            total += *in_element * sinc(i_index - index as f64);
        }
        *elem = total;
    }
}

impl eframe::App for MainData {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("I am groot");
            let less_points = [1.0,1.0,2.0,1.0,1.0];
            let mut points_out = [0.0; 34];
            let time_scale = (less_points.len() - 1) as f64 / (points_out.len() - 1) as f64;
            sinc_interpolate(&less_points, &mut points_out);
            let plot: PlotPoints = less_points.iter().enumerate().map(|a| {
                [a.0 as f64, *a.1]
            }).collect();
            let line = Line::new(plot);
            let plot2: PlotPoints = points_out.iter().enumerate().map(|a| {
                [time_scale * a.0 as f64, *a.1]
            }).collect();
            let line2 = Line::new(plot2);
            Plot::new("my_plot").view_aspect(2.0).show(ui, |plot_ui| {
                plot_ui.line(line);
                plot_ui.line(line2);
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
