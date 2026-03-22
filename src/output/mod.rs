pub mod json;
pub mod summary;
pub mod terminal;

use crate::session::state::SessionSnapshot;

/// Port: anything that can render a live `SessionSnapshot` to the user.
pub trait OutputRenderer: Send {
    /// Render or update the display with the latest snapshot.
    fn render(&mut self, snapshot: &SessionSnapshot);

    /// Called once when the session ends; print a final summary.
    fn finish(&mut self, snapshot: Option<&SessionSnapshot>);
}
