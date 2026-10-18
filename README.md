# arc_ref

Zero-copy projections of data owned by an `Arc`.

`ArcRef<O, T>` is a `&T` that keeps the owning `Arc<O>` alive. It is built by
chaining `map`, which is implemented for `Arc<O>` (the entry point) and for an
existing `ArcRef<O, T>` (to refine the view). No copying, no allocation.

```rust
use std::sync::Arc;
use arc_ref::Map;

struct Foo {
    bar: Bar,
}

struct Bar {
    value: u32,
}

let value = Arc::new(Foo { bar: Bar { value: 42 } })
    .map(|foo| &foo.bar)
    .map(|bar| &bar.value);

assert_eq!(*value, 42);
```

The target may be unsized (`str`, `[T]`, a trait object), and the owner type
only has to appear once per projection: `Arc<Foo>` -> `ArcRef<Foo, Bar>` ->
`ArcRef<Foo, u32>`. For sized targets the whole thing is two words (a pointer
plus the `Arc`).

Built on [`yoke`](https://docs.rs/yoke), using its intended `Rc`/`Arc` cart and
`map_project` support rather than a hand-rolled self-referential type.

## License

Licensed under either of MIT or Apache-2.0, at your option.
