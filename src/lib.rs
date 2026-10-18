//! Zero-copy projections of data owned by an [`Arc`].
//!
//! [`ArcRef`] holds an [`Arc<O>`](Arc) together with a borrowed view of some
//! value `T` reachable from `O`. Dereferencing an `ArcRef<O, T>` yields a `&T`,
//! exactly as if you had borrowed the field directly, while the [`Arc`] keeps
//! the storage alive for as long as the view exists.
//!
//! The view is built with [`Map::map`], which is implemented both for
//! `Arc<O>` (the entry point) and for an existing `ArcRef<O, T>` (to refine a
//! view further). Every step is a plain, zero-cost reference projection: no
//! data is copied and no allocation happens.
//!
//! The hard part is delegated to the [`yoke`](https://docs.rs/yoke) crate: an
//! `ArcRef<O, T>` is a
//! [`Yoke<&'static T, Arc<O>>`](https://docs.rs/yoke/latest/yoke/struct.Yoke.html),
//! created with
//! [`Yoke::attach_to_cart`](https://docs.rs/yoke/latest/yoke/struct.Yoke.html#method.attach_to_cart)
//! and refined with
//! [`Yoke::map_project`](https://docs.rs/yoke/latest/yoke/struct.Yoke.html#method.map_project).
//! `yoke` provides the `StableDeref`-based soundness that makes the
//! self-referential reference safe, so this crate is a thin, safe wrapper over
//! those operations and contains no `unsafe` code of its own (enforced by
//! `#![forbid(unsafe_code)]`).
//!
//! # Example
//!
//! ```
//! use std::sync::Arc;
//! use arc_ref::Map;
//!
//! struct Foo {
//!     bar: Bar,
//! }
//!
//! struct Bar {
//!     value: u32,
//! }
//!
//! let value = Arc::new(Foo { bar: Bar { value: 42 } })
//!     .map(|foo| &foo.bar)
//!     .map(|bar| &bar.value);
//!
//! assert_eq!(*value, 42);
//! ```
//!
//! Because the projection is a reference rather than a copy, the target may be
//! unsized:
//!
//! ```
//! use std::sync::Arc;
//! use arc_ref::Map;
//!
//! struct Holder {
//!     name: String,
//! }
//!
//! let holder = Arc::new(Holder { name: "hi".to_owned() });
//! let name: arc_ref::ArcRef<Holder, str> = holder.map(|h| h.name.as_str());
//!
//! assert_eq!(&*name, "hi");
//! ```
//!
//! # Size
//!
//! An `ArcRef<O, T>` is the view (`&T`) plus the owning [`Arc`]; its size does
//! not depend on the size of `O`, nor on how many times it has been mapped.
//! Measured with [`std::mem::size_of`] on a 64-bit target:
//!
//! | Type                                   | Size (bytes) |
//! |----------------------------------------|--------------|
//! | `Arc<O>`                               | 8            |
//! | `ArcRef<O, u32>` (sized `T`)           | 16           |
//! | `ArcRef<O, str>`, `ArcRef<O, [u8]>`    | 24           |
//!
//! Sized targets therefore cost two words — a thin `&T` plus the `Arc` — while
//! unsized targets (including trait objects) cost three, because the stored
//! `&T` is a fat pointer. The integer sizes scale with the pointer width, so on
//! a 32-bit target these are 8 and 12 bytes respectively.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::ops::Deref;
use std::sync::Arc;

use yoke::Yoke;

/// A borrowed view of `T` that keeps the owning [`Arc<O>`](Arc) alive.
///
/// An `ArcRef<O, T>` behaves like a `&T` (it implements [`Deref`]) but owns a
/// clone of the [`Arc`] holding `O`, so it is valid for as long as the
/// `ArcRef` itself. It is constructed and refined with [`Map::map`]:
///
/// ```
/// use std::sync::Arc;
/// use arc_ref::Map;
///
/// struct Foo {
///     count: u32,
/// }
///
/// let count = Arc::new(Foo { count: 3 }).map(|foo| &foo.count);
/// assert_eq!(*count, 3);
/// ```
///
/// Internally this is a
/// [`Yoke`](https://docs.rs/yoke/latest/yoke/struct.Yoke.html) whose cart is an
/// `Arc<O>` and whose yokeable is a `&'static T`. It is two words wide (a thin
/// `&T` plus the `Arc`) for sized `T`, and three words for unsized `T`; see the
/// [crate-level size table](crate#size) for measured values.
pub struct ArcRef<O: 'static, T: ?Sized + 'static> {
    inner: Yoke<&'static T, Arc<O>>,
}

impl<O, T> Deref for ArcRef<O, T>
where
    O: 'static,
    T: ?Sized + 'static,
{
    type Target = T;

    /// Returns the borrowed view of `T`.
    fn deref(&self) -> &T {
        self.inner.get()
    }
}

/// An ownership-preserving projection: `Self` becomes a view of `U`.
///
/// Implemented for [`Arc`] to start a projection from the root owner, and for
/// [`ArcRef`] to refine an existing view. Both forms consume `self` and move
/// (or clone) the underlying [`Arc`] into the returned [`ArcRef`].
///
/// Import the trait to call [`map`](Map::map) on either receiver:
///
/// ```
/// use std::sync::Arc;
/// use arc_ref::Map;
///
/// struct Foo {
///     inner: Inner,
/// }
///
/// struct Inner {
///     value: u32,
/// }
///
/// let value = Arc::new(Foo { inner: Inner { value: 9 } })
///     .map(|foo| &foo.inner)
///     .map(|inner| &inner.value);
///
/// assert_eq!(*value, 9);
/// ```
pub trait Map {
    /// The type stored inside the backing [`Arc`].
    type Owner: 'static;

    /// The type the current receiver borrows from.
    type Target: ?Sized + 'static;

    /// Projects a view of `Self::Target` into a view of `U`, keeping the same
    /// owner alive.
    ///
    /// `f` must only borrow from the value it is given, so the returned view
    /// stays valid for as long as the backing owner lives. This is enforced by
    /// the higher-ranked `for<'a>` bound on `f`.
    fn map<U, F>(self, f: F) -> ArcRef<Self::Owner, U>
    where
        U: ?Sized + 'static,
        F: for<'a> FnOnce(&'a Self::Target) -> &'a U;
}

impl<O> Map for Arc<O>
where
    O: 'static,
{
    type Owner = O;
    type Target = O;

    fn map<U, F>(self, f: F) -> ArcRef<O, U>
    where
        U: ?Sized + 'static,
        F: for<'a> FnOnce(&'a O) -> &'a U,
    {
        ArcRef {
            inner: Yoke::<&'static U, _>::attach_to_cart(self, f),
        }
    }
}

impl<O, T> Map for ArcRef<O, T>
where
    O: 'static,
    T: ?Sized + 'static,
{
    type Owner = O;
    type Target = T;

    fn map<U, F>(self, f: F) -> ArcRef<O, U>
    where
        U: ?Sized + 'static,
        F: for<'a> FnOnce(&'a T) -> &'a U,
    {
        ArcRef {
            inner: self.inner.map_project(|value, _| f(value)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem;
    use std::sync::{Arc, Weak};

    use super::{ArcRef, Map};

    struct Foo {
        bar: Bar,
    }

    struct Bar {
        value: u32,
    }

    struct Holder {
        name: String,
    }

    fn foo() -> Foo {
        Foo {
            bar: Bar { value: 42 },
        }
    }

    #[test]
    fn map_from_arc_to_field() {
        let value = Arc::new(foo()).map(|foo| &foo.bar.value);
        assert_eq!(*value, 42);
    }

    #[test]
    fn map_chains_through_intermediate() {
        let bar: ArcRef<Foo, Bar> = Arc::new(foo()).map(|foo| &foo.bar);
        let value: ArcRef<Foo, u32> = bar.map(|bar| &bar.value);
        assert_eq!(*value, 42);
    }

    #[test]
    fn deref_gives_plain_reference() {
        let value = Arc::new(foo()).map(|foo| &foo.bar.value);
        // Deref coercion to &u32
        let plain: &u32 = &value;
        assert_eq!(*plain, 42);
        assert_eq!(value.to_string(), "42");
    }

    #[test]
    fn unsized_target_works() {
        // `str` is the target, so the stored view is genuinely unsized.
        let holder = Arc::new(Holder {
            name: "hello".to_owned(),
        });
        let name: ArcRef<Holder, str> = holder.map(|h| h.name.as_str());
        assert_eq!(&*name, "hello");
        assert_eq!(name.len(), 5);
    }

    #[test]
    fn slice_target_works() {
        let items: ArcRef<Vec<u8>, [u8]> = Arc::new(vec![1u8, 2, 3]).map(|v| &v[..]);
        assert_eq!(&*items, &[1, 2, 3]);
    }

    #[test]
    fn keeps_owner_alive_until_dropped() {
        let owner = Arc::new(foo());
        let weak: Weak<Foo> = Arc::downgrade(&owner);

        // `map` consumes the `Arc`, moving it into the returned view.
        let value = owner.map(|foo| &foo.bar.value);

        // The ArcRef still holds a strong reference to the owner.
        assert!(weak.upgrade().is_some());
        assert_eq!(*value, 42);

        drop(value);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn chained_projection_keeps_owner_alive() {
        let owner = Arc::new(foo());
        let weak: Weak<Foo> = Arc::downgrade(&owner);

        // Two chained projections still resolve to a single owned `Arc`.
        let value = owner.map(|foo| &foo.bar).map(|bar| &bar.value);

        assert!(weak.upgrade().is_some());
        assert_eq!(*value, 42);
        drop(value);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn representation_size_matches_documented_table() {
        let word = mem::size_of::<usize>();

        // The owning `Arc` on its own.
        assert_eq!(mem::size_of::<Arc<Foo>>(), word);

        // A view of a sized target is a thin `&T` plus the `Arc`.
        assert_eq!(mem::size_of::<ArcRef<Foo, u32>>(), 2 * word);
        assert_eq!(mem::size_of::<ArcRef<Foo, Bar>>(), 2 * word);

        // Size does not grow with a deeper (more specific) owner type.
        assert_eq!(
            mem::size_of::<ArcRef<Bar, Foo>>(),
            mem::size_of::<ArcRef<Foo, u32>>()
        );

        // A view of an unsized target is a fat `&T` plus the `Arc`.
        assert_eq!(mem::size_of::<ArcRef<Foo, str>>(), 3 * word);
        assert_eq!(mem::size_of::<ArcRef<Foo, [u8]>>(), 3 * word);
    }
}
