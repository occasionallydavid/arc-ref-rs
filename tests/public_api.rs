//! Integration tests that exercise `arc_ref` through its public interface only.

use std::sync::Arc;

use arc_ref::{ArcRef, Map};

struct Config {
    server: Server,
}

struct Server {
    host: String,
    ports: Vec<u16>,
}

#[test]
fn project_nested_fields_through_public_api() {
    let config = Arc::new(Config {
        server: Server {
            host: "example.com".to_owned(),
            ports: vec![80, 443],
        },
    });

    let host: ArcRef<Config, str> = config.map(|config| config.server.host.as_str());
    assert_eq!(&*host, "example.com");
}

#[test]
fn project_unsized_slice_through_public_api() {
    let config = Arc::new(Config {
        server: Server {
            host: "example.com".to_owned(),
            ports: vec![80, 443],
        },
    });

    let ports: ArcRef<Config, [u16]> = config.map(|config| &config.server.ports[..]);
    assert_eq!(&*ports, &[80, 443]);
    assert_eq!(ports.len(), 2);
}
