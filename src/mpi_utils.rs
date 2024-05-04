use mpi::{ffi::MPI_Comm, Rank, topology::SimpleCommunicator, traits::*};
use serde::{de::DeserializeOwned, Serialize};

pub const ROOT_RANK: Rank = 0;

pub trait MPITransferable: Serialize + DeserializeOwned {
    fn into_bytes(self) -> Vec<u8> {
        serde_cbor::to_vec(&self).unwrap()
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        serde_cbor::from_slice(bytes).unwrap()
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

pub fn mpi_split_data_across_nodes<T: Default>(
    data: &[T],
    communicator: &SimpleCommunicator,
    worker_rank: Rank,
) -> Vec<T> {
    let size = communicator.size();
    let rank = communicator.rank();
    let process = communicator.process_at_rank(worker_rank);
    let split_size = data.len() / size as usize;
    assert_eq!(data.len() % size as usize, 0);

    let mut rec_data: Vec<T> = vec![T::default(); split_size];

    if rank == worker_rank {
        process.scatter_into_root(&data, &mut rec_data);
    } else {
        process.scatter_into(&mut rec_data);
    }

    rec_data
}
