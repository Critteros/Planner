use serde::Serialize;
use serde::de::DeserializeOwned;


pub trait MPITransferable: Serialize + DeserializeOwned {
    fn into_bytes(self) -> Vec<u8> {
        serde_cbor::to_vec(&self).unwrap()
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        serde_cbor::from_slice(bytes).unwrap()
    }
}

impl<T: Serialize + DeserializeOwned> MPITransferable for T {}
