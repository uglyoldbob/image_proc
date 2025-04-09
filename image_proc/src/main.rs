use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Read, Write},
    thread::JoinHandle,
    time::Duration,
};

use eframe::{
    CreationContext,
    egui::ColorImage,
};
use egui_plot::{Line, Plot, PlotPoints};
use opencv::{
    core::{MatTraitConst, MatTraitConstManual, MatTraitManual},
    videoio::VideoCaptureTrait,
};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct SaveableOpencvMat {
    width: i32,
    height: i32,
    typ: i32,
    data: Vec<u8>,
}

impl From<opencv::core::Mat> for SaveableOpencvMat {
    fn from(value: opencv::core::Mat) -> Self {
        let s = value.size().unwrap();
        let t = Self {
            width: s.width,
            height: s.height,
            typ: value.typ(),
            data: value.data_bytes().unwrap().to_vec(),
        };
        t
    }
}

impl opencv::core::MatTraitConst for SaveableOpencvMat {
    fn as_raw_Mat(&self) -> *const opencv::mod_prelude_sys::c_void {
        self.data.as_ptr() as *const opencv::mod_prelude_sys::c_void
    }
}

impl Into<opencv::core::Mat> for SaveableOpencvMat {
    fn into(self) -> opencv::core::Mat {
        let mut m = opencv::core::Mat::default();
        let mut size = opencv::core::Size::default();
        size.width = self.width;
        size.height = self.height;
        let mut orig = opencv::core::Mat::new_size_with_default(
            size,
            opencv::core::CV_64FC1,
            Default::default(),
        )
        .unwrap();
        let p = orig.data_bytes_mut().unwrap();
        p.copy_from_slice(&self.data);
        let a = orig.copy_to(&mut m);
        println!("Copy image result is {:?}", a);
        m
    }
}

#[enum_dispatch::enum_dispatch]
trait CalibrationDataTrait {
    fn apply_calibration(&self, img: ColorImage) -> ColorImage;
}

#[enum_dispatch::enum_dispatch(CalibrationDataTrait)]
#[derive(serde::Serialize, serde::Deserialize)]
enum CalibrationData {
    OpenCvCharuco([SaveableOpencvMat; 2]),
}

impl CalibrationDataTrait for [SaveableOpencvMat; 2] {
    fn apply_calibration(&self, img: ColorImage) -> ColorImage {
        println!("colorimg is {:?}", img);
        let mut m = opencv::core::Mat::default();
        let mut size = opencv::core::Size::default();
        size.width = img.width() as i32;
        size.height = img.height() as i32;
        println!("Size2 is {:?}", size);
        let mut orig = opencv::core::Mat::new_size_with_default(
            size,
            opencv::core::CV_8UC3,
            Default::default(),
        )
        .unwrap();
        let p = orig.data_bytes_mut().unwrap();
        let cdata = &img.pixels;
        let rdata: Vec<u8> = cdata
            .iter()
            .map(|a| [a.b(), a.g(), a.r()])
            .flatten()
            .collect();
        p.copy_from_slice(&rdata);
        let _ = orig.copy_to(&mut m);
        let mat = m;
        let mut oimg: opencv::core::Mat = Default::default();
        let cm: opencv::core::Mat = self[0].clone().into();
        let dc: opencv::core::Mat = self[1].clone().into();
        println!("CM: {:?}", cm);
        println!("DC: {:?}", dc);
        let a = opencv::calib3d::undistort(&mat, &mut oimg, &cm, &dc, &opencv::core::no_array());
        println!("Applied calibration {:?}", a);
        let data = oimg.data_bytes().unwrap();
        let dims = [oimg.cols() as usize, oimg.rows() as usize];
        let cimg = eframe::egui::ColorImage::from_rgb(dims, data);
        cimg
    }
}

#[derive(Debug)]
struct OpenCvCamera {
    cam: Option<opencv::videoio::VideoCapture>,
    i: i32,
    height: Option<f64>,
    width: Option<f64>,
}

enum ToCameraThread {
    ValidCamera(i32, OpenCvCamera),
    OpenCamera(i32),
    CloseCamera(i32),
    Quit,
}

enum FromCameraThread {
    CameraImage(i32, Box<opencv::core::Mat>),
}

fn live_camera_thread(
    rcv: crossbeam::channel::Receiver<ToCameraThread>,
    snd: crossbeam::channel::Sender<FromCameraThread>,
) {
    let mut live_cameras: BTreeMap<i32, OpenCvCamera> = BTreeMap::new();
    loop {
        if let Ok(a) = rcv.try_recv() {
            match a {
                ToCameraThread::ValidCamera(i, c) => {
                    live_cameras.insert(i, c);
                }
                ToCameraThread::OpenCamera(i) => {
                    if let Some(c) = live_cameras.get_mut(&i) {
                        c.open();
                    }
                }
                ToCameraThread::CloseCamera(i) => {
                    if let Some(c) = live_cameras.get_mut(&i) {
                        c.close();
                    }
                }
                ToCameraThread::Quit => {
                    break;
                }
            }
        }
        for (i, c) in &mut live_cameras {
            if c.is_open() {
                let m = c.get_image();
                if let Some(m) = m {
                    let _ = snd.send(FromCameraThread::CameraImage(*i, Box::new(m)));
                }
            }
        }
    }
}

impl OpenCvCamera {
    fn new(i: i32) -> Option<Self> {
        use opencv::videoio::VideoCaptureTraitConst;
        let mut s = Self {
            cam: None,
            i,
            height: None,
            width: None,
        };
        let mut s = if s.open() { Some(s) } else { None };
        if let Some(s) = &mut s {
            if let Some(cam) = &s.cam {
                s.width = cam
                    .get(opencv::videoio::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH as i32)
                    .ok();
                s.height = cam
                    .get(opencv::videoio::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT as i32)
                    .ok();
            }
        }
        s
    }

    fn close(&mut self) {
        self.cam = None;
    }

    fn is_open(&self) -> bool {
        self.cam.is_some()
    }

    fn get_image(&mut self) -> Option<opencv::core::Mat> {
        use opencv::videoio::VideoCaptureTrait;
        if let Some(c) = &mut self.cam {
            let mut mat = opencv::core::Mat::default();
            if let Ok(true) = c.read(&mut mat) {
                Some(mat)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn open(&mut self) -> bool {
        if self.cam.is_none() {
            if let Ok(mut c) = opencv::videoio::VideoCapture::new(self.i, opencv::videoio::CAP_ANY)
            {
                let r = c.open(self.i, opencv::videoio::CAP_ANY);
                if let Ok(true) = r {
                    self.cam = Some(c);
                    true
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            true
        }
    }
}

struct MainData {
    scale: Vec<f64>,
    actual_image: Option<eframe::egui::ColorImage>,
    img: Option<eframe::egui::TextureHandle>,
    corrected_img: Option<eframe::egui::TextureHandle>,
    live_cameras: BTreeSet<i32>,
    selected_camera: Option<i32>,
    charuco_images: Vec<opencv::core::Mat>,
    charuco_board: opencv::core::Ptr<opencv::aruco::CharucoBoard>,
    _image_thread: JoinHandle<()>,
    image_set: BTreeMap<i32, Box<opencv::core::Mat>>,
    to_image_thread: crossbeam::channel::Sender<ToCameraThread>,
    from_image_thread: crossbeam::channel::Receiver<FromCameraThread>,
    cd: Option<CalibrationData>,
    apply_cd: bool,
}

impl MainData {
    fn new(_cc: &CreationContext) -> Self {
        let to_thread = crossbeam::channel::bounded(5);
        let from_thread = crossbeam::channel::bounded(5);
        let t = std::thread::spawn(|| live_camera_thread(to_thread.1, from_thread.0));
        let cboard = make_charuco_board().unwrap();
        Self {
            scale: vec![0.0; 32],
            actual_image: None,
            img: None,
            corrected_img: None,
            live_cameras: BTreeSet::new(),
            selected_camera: None,
            charuco_images: Vec::new(),
            charuco_board: cboard,
            _image_thread: t,
            image_set: BTreeMap::new(),
            to_image_thread: to_thread.0,
            from_image_thread: from_thread.1,
            cd: None,
            apply_cd: true,
        }
    }

    fn detect_cameras(&mut self) {
        let mut consecutive_fail = 0;
        for i in 0.. {
            if let Some(mut c) = OpenCvCamera::new(i) {
                consecutive_fail = 0;
                c.close();
                let _ = self.to_image_thread.send(ToCameraThread::ValidCamera(i, c));
                self.live_cameras.insert(i);
            } else {
                consecutive_fail += 1;
            }
            if consecutive_fail == 5 {
                break;
            }
        }
        println!("Found {} cameras", self.live_cameras.len());
    }

    fn make_charuco_mat(&mut self) -> opencv::core::Mat {
        let mut pic = opencv::core::Mat::default();
        opencv::aruco::CharucoBoardTrait::draw(
            &mut self.charuco_board,
            opencv::core::Size {
                width: 2400,
                height: 2400,
            },
            &mut pic,
            10,
            1,
        )
        .unwrap();
        pic
    }

    fn save_charuco_image(&mut self) {
        println!("Saving charuco board");
        let pic = self.make_charuco_mat();
        let _ = opencv::imgcodecs::imwrite("./charuco.png", &pic, &opencv::core::Vector::new());
    }

    fn calibrate_camera(&mut self, _i: i32) -> Result<(), ()> {
        let d = get_charuco_dictionary().ok_or(())?;
        if self.charuco_images.is_empty() {
            return Err(());
        }
        let mut camera_matrix: opencv::core::Mat = Default::default();
        let mut dist_coeffs: opencv::core::Mat = Default::default();
        let mut all_corners: opencv::core::Vector<opencv::core::Vector<opencv::core::Point2f>> =
            Default::default();
        let mut all_corners_a: opencv::core::Vector<opencv::core::Point2f> = Default::default();
        let mut all_ids: opencv::core::Vector<opencv::core::Vector<i32>> = Default::default();
        let mut all_ids_a: opencv::core::Vector<i32> = Default::default();
        println!("Calibrating with {} images", self.charuco_images.len());
        for img in &self.charuco_images {
            let mut corners: opencv::core::Vector<opencv::core::Vector<opencv::core::Point2f>> =
                Default::default();
            let mut a: opencv::core::Vector<opencv::core::Point2f> = Default::default();
            a.push(Default::default());
            a.push(Default::default());
            a.push(Default::default());
            a.push(Default::default());
            let num_things = 81;
            for _ in 0..num_things {
                corners.push(a.clone());
            }
            let mut ids: opencv::core::Vector<i32> = Default::default();
            for _ in 0..num_things {
                ids.push(0);
            }
            let a = opencv::aruco::detect_markers_def(&img, &d, &mut corners, &mut ids);
            for r in &corners {
                println!("Accepted corner: {:?}", r);
            }
            println!("There are {} corners, {} ids", corners.len(), ids.len());
            for id in &ids {
                println!("ID: {}", id);
            }
            if a.is_ok() {
                println!("Detect markers is ok");
                let mut charuco_corners: opencv::core::Mat = Default::default();
                let mut charuco_ids: opencv::core::Mat = Default::default();
                let b = opencv::aruco::interpolate_corners_charuco_def(
                    &corners,
                    &ids,
                    img,
                    &self.charuco_board,
                    &mut charuco_corners,
                    &mut charuco_ids,
                )
                .map_err(|_| ())?;
                println!("Interpolate retured {}", b);
                println!(
                    "charuco: There are {} corners, {} ids",
                    charuco_corners.rows(),
                    charuco_ids.rows()
                );
                println!(
                    "Charuco corner element size {:?}",
                    charuco_corners.elem_size()
                );
                let cc: Vec<Vec<opencv::core::Point2f>> = charuco_corners.to_vec_2d().unwrap();
                for i in cc {
                    println!("charuco corner: {:?}", i);
                    all_corners_a.push(i[0]);
                }
                let cc: Vec<Vec<i32>> = charuco_ids.to_vec_2d().unwrap();
                for i in cc {
                    println!("charuco id: {:?}", i);
                    all_ids_a.push(i[0]);
                }
            }
        }
        all_corners.push(all_corners_a);
        all_ids.push(all_ids_a);
        let criteria = opencv::core::TermCriteria {
            typ: opencv::core::TermCriteria_Type::EPS as i32
                + opencv::core::TermCriteria_Type::COUNT as i32,
            max_count: 30,
            epsilon: 0.1,
        };
        let size = opencv::core::Size {
            width: self.charuco_images[0].cols(),
            height: self.charuco_images[0].rows(),
        };
        println!("Size is {:?}", size);
        let c = opencv::aruco::calibrate_camera_charuco(
            &all_corners,
            &all_ids,
            &self.charuco_board,
            size,
            &mut camera_matrix,
            &mut dist_coeffs,
            &mut opencv::core::no_array(),
            &mut opencv::core::no_array(),
            0,
            criteria,
        );
        println!(
            "Calibrate returned {:?} {:?} {:?}",
            c, camera_matrix, dist_coeffs
        );
        let cm: SaveableOpencvMat = camera_matrix.into();
        let dc: SaveableOpencvMat = dist_coeffs.into();
        let cd = CalibrationData::OpenCvCharuco([cm, dc]);
        let data = bincode::serde::encode_to_vec(&cd, bincode::config::standard());
        if let Ok(data) = data {
            let mut f = std::fs::File::create("./test.bin").unwrap();
            f.write_all(&data).unwrap();
        }
        self.cd = Some(cd);
        Ok(())
    }

    fn check_charuco_image(
        &self,
        img: &opencv::core::Mat,
        debug: Option<&mut opencv::core::Mat>,
    ) -> i32 {
        if let Some(d) = get_charuco_dictionary() {
            let mut corners: opencv::core::Vector<opencv::core::Vector<opencv::core::Point2f>> =
                Default::default();
            let mut a: opencv::core::Vector<opencv::core::Point2f> = Default::default();
            a.push(opencv::core::Point2f::new(1.0, 2.0));
            a.push(opencv::core::Point2f::new(1.0, 2.0));
            a.push(opencv::core::Point2f::new(1.0, 2.0));
            a.push(opencv::core::Point2f::new(1.0, 2.0));
            let num_things = 81;
            for _ in 0..num_things {
                corners.push(a.clone());
            }
            let mut ids: opencv::core::Vector<i32> = Default::default();
            for _ in 0..num_things {
                ids.push(1);
            }
            let mut rejected: opencv::core::Vector<opencv::core::Vector<opencv::core::Point2f>> =
                Default::default();
            let mut a: opencv::core::Vector<opencv::core::Point2f> = Default::default();
            a.push(opencv::core::Point2f::new(1.0, 2.0));
            a.push(opencv::core::Point2f::new(1.0, 2.0));
            a.push(opencv::core::Point2f::new(1.0, 2.0));
            a.push(opencv::core::Point2f::new(1.0, 2.0));
            for _ in 0..num_things {
                rejected.push(a.clone());
            }
            if debug.is_some() {
                for r in &rejected {
                    println!("Rejected point {:?}", r);
                }
                for r in &corners {
                    println!("Accepted corner: {:?}", r);
                }
                for i in &ids {
                    println!("ID {}", i)
                }
            }
            let a = opencv::aruco::detect_markers(
                img,
                &d,
                &mut corners,
                &mut ids,
                &opencv::aruco::DetectorParameters::create().unwrap(),
                &mut rejected,
            );
            if debug.is_some() {
                println!("Detect markers: {:?} {} {}", a, corners.len(), ids.len());
            }
            if a.is_ok() {
                let mut charuco_corners: opencv::core::Mat = Default::default();
                let mut charuco_ids: opencv::core::Mat = Default::default();
                let mut vimgs: opencv::core::Vector<opencv::core::Mat> =
                    opencv::core::Vector::new();
                vimgs.push(img.to_owned());
                let b = opencv::aruco::interpolate_corners_charuco_def(
                    &corners,
                    &ids,
                    img,
                    &self.charuco_board,
                    &mut charuco_corners,
                    &mut charuco_ids,
                );
                if debug.is_some() {
                    println!("interpolate is {:?}", b);
                }
                if b.is_ok() {
                    if let Some(debug) = debug {
                        let test = opencv::aruco::draw_detected_corners_charuco(
                            debug,
                            &charuco_corners,
                            &opencv::core::no_array(),
                            opencv::core::VecN([255.0, 0.0, 0.0, 255.0]),
                        );
                        println!("Test is {:?}", test);
                        println!("Charuco corners channels {}", debug.channels());
                        let asdf = opencv::imgcodecs::imwrite(
                            "./charuco_corners.png",
                            debug,
                            &opencv::core::Vector::new(),
                        );
                        println!("Result of saving charuco corners {:?}", asdf);
                    }
                }
                if let Ok(b) = b { b } else { 0 }
            } else {
                0
            }
        } else {
            0
        }
    }
}

fn get_charuco_dictionary() -> Option<opencv::core::Ptr<opencv::aruco::Dictionary>> {
    let dict = opencv::aruco::DICT_6X6_1000;
    let d = opencv::aruco::Dictionary::get(dict);
    d.ok()
}

fn make_charuco_board() -> Option<opencv::core::Ptr<opencv::aruco::CharucoBoard>> {
    if let Some(d) = get_charuco_dictionary() {
        println!("Making charuco board");
        let board = opencv::aruco::CharucoBoard::create(10, 10, 10.0 * 0.0254, 7.0 * 0.0254, &d);
        board.ok()
    } else {
        None
    }
}

impl eframe::App for MainData {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        let _ = self.to_image_thread.send(ToCameraThread::Quit);
    }

    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(Duration::from_millis(10));
        let mut use_newest_image = false;
        while let Ok(a) = self.from_image_thread.try_recv() {
            match a {
                FromCameraThread::CameraImage(i, bm) => {
                    if let Some(j) = self.selected_camera {
                        if j != i {
                            let _ = self.to_image_thread.send(ToCameraThread::CloseCamera(i));
                        }
                    }
                    self.image_set.insert(i, bm);
                }
            }
        }
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            egui_extras::install_image_loaders(ctx);

            eframe::egui::ScrollArea::vertical().show(ui, |ui| {
                ui.horizontal(|ui| {
                    eframe::egui::ComboBox::from_label("Select a camera")
                        .selected_text(format!("{:?}", self.selected_camera))
                        .show_ui(ui, |ui| {
                            for i in &self.live_cameras {
                                ui.selectable_value(
                                    &mut self.selected_camera,
                                    Some(*i),
                                    format!("Camera {}", i),
                                );
                            }
                        });
                    if ui.button("Open camera").clicked() {
                        if let Some(i) = self.selected_camera {
                            let _ = self.to_image_thread.send(ToCameraThread::OpenCamera(i));
                        }
                    }
                    if ui.button("Close camera").clicked() {
                        if let Some(i) = self.selected_camera {
                            let _ = self.to_image_thread.send(ToCameraThread::CloseCamera(i));
                        }
                    }
                });
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
                    if ui.button("Generate charuco pattern").clicked() {
                        self.save_charuco_image();
                    }
                    if ui.button("Save charuco capture from camera").clicked() {
                        use_newest_image = true;
                    }
                    if ui.button("Use charuco mat directly").clicked() {
                        let m = self.make_charuco_mat();
                        self.charuco_images.push(m);
                    }
                    if ui.button("Clear saved images").clicked() {
                        self.charuco_images.clear();
                    }
                    if ui.button("Do calibration").clicked() {
                        if let Some(i) = self.selected_camera {
                            let _ = self.calibrate_camera(i);
                        }
                    }
                });
                if ui.button("Debug1").clicked() {
                    let m = Box::new(self.make_charuco_mat());
                    let mut newmat = self.make_charuco_mat();
                    self.check_charuco_image(&m, Some(&mut newmat));
                    let data = newmat.data_bytes().unwrap();
                    let dims = [newmat.cols() as usize, newmat.rows() as usize];
                    let data: Vec<u8> = data.iter().map(|a| [*a, *a, *a]).flatten().collect();
                    let cimg = eframe::egui::ColorImage::from_rgb(dims, &data);
                    let a = ctx.load_texture(
                        "actual_image",
                        cimg.clone(),
                        eframe::egui::TextureOptions::LINEAR,
                    );
                    self.actual_image.replace(cimg);
                    self.img.replace(a);
                }
                ui.checkbox(&mut self.apply_cd, "Apply calibration");
                ui.label(format!(
                    "There are {} saved charuco images",
                    self.charuco_images.len()
                ));
                if let Some(i) = &self.selected_camera {
                    if let Some(img) = self.image_set.get(i) {
                        if use_newest_image {
                            self.charuco_images.push(*img.clone());
                        }
                        if let Some(cd) = &self.cd {
                            if let Ok(data) = img.data_bytes() {
                                let dims = [img.cols() as usize, img.rows() as usize];
                                let egui_img = eframe::egui::ColorImage::from_rgb(dims, data);
                                let cimg = cd.apply_calibration(egui_img);
                                let a = ctx.load_texture(
                                    "actual_image",
                                    cimg.clone(),
                                    eframe::egui::TextureOptions::LINEAR,
                                );
                                self.actual_image.replace(cimg);
                                self.img.replace(a);
                            }
                        } else {
                            if let Ok(data) = img.data_bytes() {
                                let dims = [img.cols() as usize, img.rows() as usize];
                                let cimg = eframe::egui::ColorImage::from_rgb(dims, data);
                                let a = ctx.load_texture(
                                    "actual_image",
                                    cimg.clone(),
                                    eframe::egui::TextureOptions::LINEAR,
                                );
                                self.actual_image.replace(cimg);
                                self.img.replace(a);
                            }
                        }
                    }
                }
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
        Box::new(|cc| {
            let mut md = MainData::new(cc);
            md.detect_cameras();
            Ok(Box::new(md))
        }),
    )
    .unwrap();
}
