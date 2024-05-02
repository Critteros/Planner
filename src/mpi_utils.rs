use mpi::Rank;
use mpi::topology::{SimpleCommunicator};
use mpi::traits::*;
use crate::algorithm::data::MPITransferable;

pub fn mpi_synchronize_ref<T: MPITransferable + Clone>(
    variable: &mut T,
    communicator: &SimpleCommunicator,
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