//     #[test]
//     fn computing_party_init_runtime_context() {
        
//         use rustlynx::computing_party::init;
//         use std::net::{IpAddr, Ipv4Addr};
//         use std::net::{TcpStream, TcpListener, SocketAddr};
//         use std::env;
//         // args should contain testing/cfg/{Party0.toml,Party1.toml}

//         let test_cfg = env::args().collect();
//         let mut ctx = init::runtime_context( &test_cfg ).unwrap();


//         assert!( init::connection( &mut ctx ).is_ok() );

//         // assert_eq!(
//         //     SocketAddr::new(IpAddr::V4(ctx.network.local.ip.clone()), ctx.network.local.port), 
//         //     ctx.network.external.tcp.as_ref().unwrap().i_stream.local_addr().unwrap()
//         // );

// }