use crate::MigrateArgs;
use anyhow::Result;
use tangram_app_core::default_database_url;

pub fn migrate(args: MigrateArgs) -> Result<()> {
	let database_url = match args.database_url {
		Some(database_url) => database_url.parse()?,
		None => default_database_url(),
	};
	tangram_app_core::migrate(database_url)
}
