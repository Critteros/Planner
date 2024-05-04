use mpi::{ffi::MPI_Comm, traits::*, Rank};
use rayon::prelude::*;
use serde::{de::DeserializeOwned, Serialize};

/// Rank of the root process (data owner)
pub const ROOT_RANK: Rank = 0;

/// Trait for types that can be transferred over MPI as bytes
///
/// Utilizes bincode instead of serde_cbor because cbor
/// 'compresses' the data. For example int field can have different size when serialized.
/// In our case, we need to have the same size of data on all nodes to be able to scatter
/// and gather it.
pub trait MPITransferable: Serialize + DeserializeOwned {
    /// Serialize the object into a byte vector
    fn into_bytes(self) -> Vec<u8> {
        bincode::serialize(&self).unwrap()
    }
    
    /// Deserialize the object from a byte vector
    fn from_bytes(bytes: &[u8]) -> Self {
        bincode::deserialize(bytes).unwrap()
    }
}

impl<T: Serialize + DeserializeOwned> MPITransferable for T {}

/// Synchronize a variable between all processes
pub fn mpi_synchronize_ref<T: MPITransferable + Clone>(
    variable: &mut T,
    communicator: &impl Communicator<Raw = MPI_Comm>,
    data_owner_rank: Rank,
) {
    let data_owner_process = communicator.process_at_rank(data_owner_rank);
    let current_rank = communicator.rank();
    let mut serialized_data_len;
    let mut serialized_data;

    if current_rank == data_owner_rank {
        serialized_data = variable.clone().into_bytes();

        serialized_data_len = serialized_data.len();
        data_owner_process.broadcast_into(&mut serialized_data_len);
        data_owner_process.broadcast_into(&mut serialized_data);
    } else {
        serialized_data_len = 0;
        data_owner_process.broadcast_into(&mut serialized_data_len);
        serialized_data = vec![0; serialized_data_len];
        data_owner_process.broadcast_into(&mut serialized_data);

        *variable = T::from_bytes(&serialized_data);
    }
}

/// Execute a function on a specific rank and synchronize the result with all
pub fn mpi_execute_and_synchronize_at<F, R>(
    f: F,
    communicator: &impl Communicator<Raw = MPI_Comm>,
    executor_rank: Rank,
) -> R
where
    F: FnOnce() -> R,
    R: MPITransferable + Clone + Default,
{
    let current_rank = communicator.rank();
    let mut value_placeholder = if current_rank == executor_rank {
        f()
    } else {
        R::default()
    };

    mpi_synchronize_ref(&mut value_placeholder, communicator, executor_rank);
    return value_placeholder;
}

/// Serialize a vector of MPITransferable objects into a single byte vector
///
/// Helper method for [`mpi_split_data_across_nodes`] and [`mpi_gather_and_synchronize`]
fn serialize_vec<T: Default + MPITransferable + Clone + Send>(data: Vec<T>) -> (usize, Vec<u8>) {
    let serialized_data: Vec<Vec<u8>> = data.into_par_iter().map(|x| x.into_bytes()).collect();

    // Assert that the data is the same size on all nodes
    for i in 1..serialized_data.len() {
        let current_len = serialized_data[i].len();
        let prev_len = serialized_data[i - 1].len();
        assert_eq!(prev_len, current_len);
    }

    let data_size = serialized_data[0].len();
    let serialized_data: Vec<u8> = serialized_data.into_iter().flatten().collect();

    (data_size, serialized_data)
}

/// Split data in a vector across all nodes evenly
///
/// Expects `T` elements to be the same size when serialized
pub fn mpi_split_data_across_nodes<T: Default + MPITransferable + Clone + Send>(
    data: &[T],
    communicator: &impl Communicator<Raw = MPI_Comm>,
    data_owner_rank: Rank,
) -> Vec<T> {
    assert_ne!(data.len(), 0);
    let size = communicator.size();
    let rank = communicator.rank();
    let process = communicator.process_at_rank(data_owner_rank);
    let split_size = data.len() / size as usize;
    assert_eq!(data.len() % size as usize, 0);

    let mut rec_data: Vec<u8>;
    let mut data_size = 0;

    if rank == data_owner_rank {
        let serialized_data: Vec<u8>;

        (data_size, serialized_data) = serialize_vec(data.to_owned());

        mpi_synchronize_ref(&mut data_size, communicator, data_owner_rank);
        rec_data = vec![0; data_size * split_size];

        process.scatter_into_root(&serialized_data, &mut rec_data);
    } else {
        mpi_synchronize_ref(&mut data_size, communicator, data_owner_rank);
        rec_data = vec![0; data_size * split_size];
        process.scatter_into(&mut rec_data);
    }

    rec_data
        .chunks(data_size)
        .map(|chunk| T::from_bytes(chunk))
        .collect()
}

/// Gather data (shards of split data) from all nodes into a single vector
///
/// Expects `T` elements to be the same size when serialized
pub fn mpi_gather_and_synchronize<T: Default + MPITransferable + Clone + Send>(
    gather_from: &[T],
    communicator: &impl Communicator<Raw = MPI_Comm>,
    data_owner_rank: Rank,
) -> Vec<T> {
    assert_ne!(gather_from.len(), 0);
    let rank = communicator.rank();
    let process = communicator.process_at_rank(data_owner_rank);

    let mut gathered_data = Vec::new();

    let (data_size, serialized_data) = serialize_vec(gather_from.to_owned());

    if rank == data_owner_rank {
        let mut buffer: Vec<u8> = vec![0; serialized_data.len() * communicator.size() as usize];
        process.gather_into_root(&serialized_data, &mut buffer);
        gathered_data = buffer
            .chunks(data_size)
            .map(|chunk| T::from_bytes(chunk))
            .collect();
    } else {
        process.gather_into(&serialized_data);
    }

    mpi_synchronize_ref(&mut gathered_data, communicator, data_owner_rank);
    gathered_data
}
