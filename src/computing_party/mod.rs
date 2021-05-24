use std::error::Error;
use std::net::{Ipv4Addr, TcpStream};
use std::fmt;

pub mod protocol;
pub mod init;

#[allow(unused_variables, unused_mut, dead_code)]
pub mod ml;

pub fn run( cfg_file : String ) -> Result<(), Box<dyn Error>> {

	let mut ctx = 
		init::runtime_context( &cfg_file )?;
		init::connection( &mut ctx )?;

	println!("rustlynx::computing_party::run: runtime context initialized\n{}", &ctx);
	match ctx.ml.callable {
		Some(callable) => callable(&mut ctx),
		_ => Err("unexpected failure while calling ml model".into()),
	}
}

#[derive(Debug, Default, Clone)]
pub struct Context {
	pub sys: System,
	pub net: Network,
	pub num: Numeric,
	pub ml: MachineLearning,
	pub cr: CorrelatedRandomness,
}


#[derive(Debug, Default, Clone)]
pub struct System {
	pub threads: Threading,
}

#[derive(Debug, Default, Clone)]
pub struct Threading {
	pub optimise: bool,
	pub offline: usize,
	pub online: usize,
}

#[derive(Debug, Default, Clone)]
pub struct CorrelatedRandomness {
	pub debug: bool,
	pub additive: Option<Vec<(u64, u64, u64)>>,
	pub xor: Option<Vec<(u128, u128, u128)>>,
}

#[derive(Debug, Default, Clone)]
pub struct Numeric {
	pub precision_frac: usize,
	pub precision_int: usize,
	pub asymm: u64,
}

#[derive(Debug, Default, Clone)]
pub struct Network {
	pub local: User,
	pub external: User,
	pub ti: Option<User>,
}

#[derive(Debug, Default, Clone)]
pub struct User {
	pub id: u8,
	pub ip: Option<Ipv4Addr>,
	pub portrange: (u16, u16),
	pub tcp: Option<Vec<IoStream>>,
}

#[derive(Debug)]
pub struct IoStream {
	pub istream : TcpStream,
	pub ostream : TcpStream,
}

#[derive(Default, Clone)]
pub struct MachineLearning {
	pub model: Option<MLModel>,
	pub phase: Option<MLPhase>,
	pub cfg: String,
	pub callable: Option<fn(&mut Context) -> Result<(), Box<dyn Error>>>
}

#[derive(Debug, Clone)]
pub enum MLModel {
	LogisticRegression,
	NaiveBayes,
	ExtraTrees,
	RandomForest,
	TimeTest
}

#[derive(Debug, Clone)]
pub enum MLPhase {
	Learning,
	Inference,
}

impl Clone for IoStream {
    fn clone(&self) -> Self {
       IoStream {
            istream: self.istream.try_clone().unwrap(),
            ostream: self.ostream.try_clone().unwrap(),
        }
    }
}

impl fmt::Debug for MachineLearning {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "MachineLearning {{ model: {:?}, phase: {:?}, cfg: {}, callable: {} }}",
			self.model, self.phase, self.cfg, self.callable.is_some())
	}
}

impl fmt::Display for Context {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		
		write!(f, "rustlynx::computing_party::Context:
	sys.......:{:?}
	net.local.:{}
	net.extern:{}
	net.ti....:{:?}
	num.......:{:?}
	ml........:{:?}
	cr........:{:?}",
			self.sys, self.net.local, self.net.external, self.net.ti, self.num, self.ml, self.cr)	
	}		
}

impl fmt::Display for User {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		
		write!(f, "User {{ id: {}, ip: {:?}, portrange: {:?}, tcp: {} }}",
			self.id, self.ip, self.portrange, self.tcp.is_some())	
	}		
}


