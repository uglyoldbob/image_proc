pub struct Widget1 {}

pub struct Widget1State {}

impl eframe::egui::WidgetWithState for Widget1 {
    type State = Widget1State;
}

impl Widget1 {
    pub fn new() -> Self {
        Self {}
    }
}
