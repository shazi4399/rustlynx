extern crate config;
// use std::num::Wrapping;
use std::error::Error;
use std::net::{IpAddr, Ipv4Addr, TcpStream, TcpListener, SocketAddr};
use std::thread;
use std::time::Duration;
use super::super::constants;
use super::{MLModel, MLPhase, MachineLearning, Context, System, Threading, 
	CorrelatedRandomness, Numeric, Network, User, IoStream};

pub fn runtime_context(cfg_file: &String) -> Result<Context, Box<dyn Error>>  {

	let mut settings = config::Config::default();
    settings
        .merge(config::File::with_name(cfg_file.as_str())).unwrap()
        .merge(config::Environment::with_prefix("APP")).unwrap();

    /* local network settings */
    let local_id: String = settings.get_str("net.local.party_id")?;
	let local_ip: String = settings.get_str("net.local.ip")?;
	let local_portrange: String = settings.get_str("net.local.portrange")?;
    /* extern network settings */
    let extern_id: String = settings.get_str("net.extern.party_id")?;
    let extern_ip: String = settings.get_str("net.extern.ip")?;
    let extern_portrange: String = settings.get_str("net.extern.portrange")?;
	/* client-side ti settings */
	let ti_ip: String = settings.get_str("net.ti.ip")?;
	let ti_port: String = settings.get_str("net.ti.port")?;
	/* system settings */
	let optimise_threads: String = settings.get_str("sys.threads.optimise")?;
	let offline_threads: String = settings.get_str("sys.threads.offline")?;
	let online_threads: String = settings.get_str("sys.threads.offline")?;
	/* number system settings */
    let precision_frac: String = settings.get_str("numbersystem.precision_frac")?;
    let precision_int: String = settings.get_str("numbersystem.precision_int")?;
	/* correlated randomness settings */
    let debug_cr: String = settings.get_str("correlatedrandomness.debug")?;
	/* machine learning settings */
	let phase: String = settings.get_str("ml.phase")?;
	let model: String = settings.get_str("ml.model")?;

	let local_user: User = parse_user_settings(&local_id, &local_ip, &local_portrange)?;
	let extern_user: User = parse_user_settings(&extern_id, &extern_ip, &extern_portrange)?;
	let ti: Option<User> = parse_ti_settings(&ti_ip, &ti_port, &debug_cr)?;
	let ml: MachineLearning = parse_ml_settings(&phase, &model, &settings)?;
	let num: Numeric = parse_numeric_settings(&precision_int, &precision_frac, &local_id)?;
	let threading: Threading = parse_thread_settings(&optimise_threads, &offline_threads, &online_threads)?;
	let cr: CorrelatedRandomness = parse_cr_settings(&debug_cr)?;

	let sys = System { threads: threading };	
	let net = Network { local: local_user, external: extern_user, ti: ti }; 

	println!("rustlynx::computing_party::init: done parsing cfg");

	Ok(Context { sys: sys, net: net, num: num, ml: ml, cr: cr })
}

fn parse_user_settings(id: &String, ip: &String, portrange: &String) -> Result<User, Box<dyn Error>> {

	let id = id.parse::<u8>()?;
	let ip = ip.parse::<Ipv4Addr>()?;

	let portrange = (*portrange).split(':').collect::<Vec<&str>>();
	let portrange = ( portrange[0].parse::<u16>()?, portrange[1].parse::<u16>()? ); 

	Ok(User { id: id, ip: Some(ip), portrange: portrange, tcp: None })
}

fn parse_ti_settings(ip: &String, port: &String, null_ti: &String) -> Result<Option<User>, Box<dyn Error>> {

	if null_ti.parse::<bool>()? {
		return Ok(None)
	}

	let lb = port.parse::<u16>()?;
	let ub = lb + 1;
	let portrange = format!("{}:{}", lb, ub);

	Ok(Some( parse_user_settings(&String::from("255"), &ip, &portrange)? ))
}

fn parse_ml_settings(phase: &String, model: &String, settings: &config::Config) -> Result<MachineLearning, Box<dyn Error>> {

	let ml_model: Option<MLModel>;
	let ml_phase: Option<MLPhase>;
	let callable: Option<fn(&mut Context) -> Result<(), Box<dyn Error>>>;
	let cfg: String;

	if model == "logisticregression" {
		ml_model = Some(MLModel::LogisticRegression);
		if phase == "learning" {
			ml_phase = Some(MLPhase::Learning);
			cfg = settings.get_str("ml.logisticregression.learning_cfg")?;
			callable = Some(super::ml::logistic_regression::learning::run);
		} else if phase == "inference" {
			ml_phase = Some(MLPhase::Inference);
			cfg = settings.get_str("ml.logisticregression.inference_cfg")?;
			callable = Some(super::ml::logistic_regression::inference::run);
		} else {
			return Err("invalid ml.phase".into())
		}
	} else if model == "naivebayes" {
		ml_model = Some(MLModel::NaiveBayes);
		if phase == "learning" {
			ml_phase = Some(MLPhase::Learning);
			cfg = settings.get_str("ml.naivebayes.learning_cfg")?;
			callable = Some(super::ml::naive_bayes::learning::run);
		} else if phase == "inference" {
			ml_phase = Some(MLPhase::Inference);
			cfg = settings.get_str("ml.naivebayes.inference_cfg")?;
			callable = Some(super::ml::naive_bayes::inference::run);
		} else {
			return Err("invalid ml.phase".into())
		}	
	} else if model == "extratrees" {
		ml_model = Some(MLModel::ExtraTrees);
		if phase == "learning" {
			ml_phase = Some(MLPhase::Learning);
			cfg = settings.get_str("ml.extratrees.learning_cfg")?;
			callable = Some(super::ml::decision_trees::extra_trees::learning::run);
		} else if phase == "inference" {
			ml_phase = Some(MLPhase::Inference);
			cfg = settings.get_str("ml.extratrees.inference_cfg")?;
			callable = Some(super::ml::decision_trees::extra_trees::inference::run);
		} else {
			return Err("invalid ml.phase".into())
		}	
	} else if model == "randomforest" {
		ml_model = Some(MLModel::RandomForest);
		if phase == "learning" {
			ml_phase = Some(MLPhase::Learning);
			cfg = settings.get_str("ml.randomforest.learning_cfg")?;
			callable = Some(super::ml::decision_trees::random_forest::learning::run);
		} else if phase == "inference" {
			ml_phase = Some(MLPhase::Inference);
			cfg = settings.get_str("ml.randomforest.inference_cfg")?;
			callable = Some(super::ml::decision_trees::random_forest::inference::run);
		} else {
			return Err("invalid ml.phase".into())
		}	
	} else if model == "time_test" {
		ml_model = Some(MLModel::TimeTest);
		if phase == "learning" {
			ml_phase = Some(MLPhase::Learning);
			cfg = settings.get_str("ml.time_test.learning_cfg")?;
			callable = Some(super::ml::decision_trees::test_DT_protocols::protocols::test_protocol);
		} else {
			return Err("invalid ml.phase".into())
		}	
	}else {
		return Err("invalid or unimplemented ml.model".into())
	}

	Ok(MachineLearning { phase: ml_phase, model: ml_model, cfg: cfg, callable: callable })
}

fn parse_numeric_settings(precision_int: &String, precision_frac: &String, asymm: &String) -> Result<Numeric, Box<dyn Error>> {

	let precision_int = precision_int.parse::<usize>()?;
	let precision_frac = precision_frac.parse::<usize>()?;
	let asymm = asymm.parse::<u64>()?;

    if precision_int + precision_frac > constants::BITLENGTH {
    	return Err("(fractional precision + integer precision) cannot be larger than bitlength".into());
    } else if 2*(precision_int + precision_frac) > constants::BITLENGTH {
    	println!("WARNING: '2*(fractional precision + integer precision) > bitlength' may introduce truncation errors.");
    }

	Ok(Numeric {precision_int: precision_int, precision_frac: precision_frac, asymm: asymm})
}

fn parse_thread_settings(optimise: &String, offline: &String, online: &String) -> Result<Threading, Box<dyn Error>> {
	
	let optimise = optimise.parse::<bool>()?;
	let offline = offline.parse::<usize>()?;
	let online = online.parse::<usize>()?;

	Ok(Threading {optimise: optimise, offline: offline, online: online})
}

fn parse_cr_settings(debug: &String) -> Result<CorrelatedRandomness, Box<dyn Error>> {

	let debug = debug.parse::<bool>()?;

	Ok(CorrelatedRandomness {debug: debug, additive: None, xor: None})
}

pub fn connection(ctx: &mut Context) -> Result<(), Box<dyn Error>> {

	let local_ip = match ctx.net.local.ip {
		Some(ip) => ip,
		None => return Err("invalid local ip address".into())
	};

	let extern_ip = match ctx.net.external.ip {
		Some(ip) => ip,
		None => return Err("invalid extern ip address".into())
	};

	let mut server_s_addrs = Vec::new();
	let mut client_s_addrs = Vec::new();
	let mut listeners = Vec::new();

	for i in 0..ctx.sys.threads.online {
		
		server_s_addrs.push( 
			SocketAddr::new(IpAddr::V4(local_ip), ctx.net.local.portrange.0 + i as u16)
		);

		listeners.push(TcpListener::bind(&server_s_addrs[i])?);
		
		client_s_addrs.push( 
			SocketAddr::new(IpAddr::V4(extern_ip), ctx.net.external.portrange.0 + i as u16)
		);	
	}

	let mut istreams: Vec<TcpStream> = Vec::new();
	let mut ostreams: Vec<TcpStream> = Vec::new();
	
	for i in 0..ctx.sys.threads.online {

		let listener = listeners[i].try_clone().unwrap();
		let client_s_addr = client_s_addrs[i].clone();

		let istream_handle = thread::spawn(move || { 

			loop {
				match listener.accept() {
					Ok((stream, _addr)) => {
						// println!("rustlynx::computing_party::init::connection:thread{}: TcpListener {:?} established connection with {:?}",
						// 	i, &listener.local_addr(), &addr);
						return stream
					},
					Err(_) => continue,
				}
			}	
		});

		let ostream_handle = thread::spawn(move || {

			loop {
				match TcpStream::connect(client_s_addr) {
					Ok(stream) => {
						// println!("rustlynx::computing_party::init::connection:thread{}: TcpClient {:?} established connection with {:?}",
						// i, &client_s_addr, &stream.local_addr());
						return stream
					},
					Err(_) => { 
						println!("rustlynx::computing_party::init::thread{}: connection refused by {} -- retrying.", i, &client_s_addr);
						thread::sleep(Duration::from_secs(1)); 
					},
				};
			}
		});

		istreams.push(istream_handle.join().unwrap());
		ostreams.push(ostream_handle.join().unwrap());
	}

	println!("rustlynx::computing_party::init::connection: {} sockets connected", ctx.sys.threads.online);

	let mut tcp :Vec<IoStream> = Vec::new();
	for i in 0..ctx.sys.threads.online {

		let iostream = IoStream { 
			istream: istreams[i].try_clone().unwrap(), 
			ostream: ostreams[i].try_clone().unwrap() 
		};
		tcp.push(iostream)
	}
	ctx.net.external.tcp = Some(tcp);

	Ok(())
}




	// let ti_s_addr = if ctx.net.ti.is_some() {
	// 	Some(SocketAddr::new(
	// 		IpAddr::V4(ctx.net.ti.as_ref().unwrap().ip.as_ref().unwrap().clone()), 
	// 		ctx.net.ti.as_ref().unwrap().portrange.0))
	// } else {
	// 	None
	// };

	// let ti_o_stream_handle = if ctx.network.ti.is_some() {
	// 	Some(thread::spawn(move || { try_connect(&ti_s_addr.unwrap()) } ))
	// } else {
	// 	None
	// };

	// ctx.network.external.tcp = Some(IoStream{ 
	// 	i_stream: i_stream_handle.join().unwrap(),
	// 	o_stream: o_stream_handle.join().unwrap() });

	// if ctx.network.ti.is_some() {

	// 	let stream = ti_o_stream_handle.unwrap().join().unwrap();
	// 	ctx.network.ti.as_mut().unwrap().tcp = Some(IoStream {
	// 		o_stream: stream.try_clone().unwrap(),
	// 		i_stream: stream.try_clone().unwrap()
	// 	});
	// }
