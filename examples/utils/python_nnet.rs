use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use alphazero_rs::nnet::*;
use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

pub trait PythonModel {
    const SOURCE: &'static str;
    const PSEUDO_FILE_NAME: &'static str;
    const MODULE_NAME: &'static str;
}

pub struct PythonNNet<M> {
    phantom: PhantomData<M>,
    module: Py<PyModule>,
    model: PyObject,
    checkpoint: PathBuf,
}

impl<M> NNet for PythonNNet<M>
where
    M: PythonModel,
{
    fn new<P: AsRef<Path>>(checkpoint: P) -> Self {
        // get the Python interpreter initialized asap
        let gil = Python::acquire_gil();
        let py = gil.python();
        let module =
            PyModule::from_code(py, M::SOURCE, M::PSEUDO_FILE_NAME, M::MODULE_NAME).unwrap();

        let model: PyObject = module.call0("init_model").unwrap().to_object(py);

        PythonNNet {
            phantom: PhantomData,
            module: module.into(),
            model: model,
            checkpoint: checkpoint.as_ref().to_path_buf(),
        }
    }

    fn train(&mut self, examples: SOATrainingSamples, previous_model_id: usize, model_id: usize) {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let examples: [PyObject; 3] = [
            examples.0.to_owned().into_pyarray(py).into(),
            examples.1.to_owned().into_pyarray(py).into(),
            examples.2.to_owned().into_pyarray(py).into(),
        ];
        let examples = PyTuple::new(py, &examples);

        let module = self.module.as_ref(py);
        let new_model = module
            .call("train_model", (self.model.as_ref(py), examples), None)
            .unwrap();
    }

    fn predict(
        &self,
        board: BatchedBoardFeaturesView,
        model_id: usize,
    ) -> (BatchedPolicy, BatchedValue) {
        (BatchedPolicy::zeros([3, 3]), BatchedValue::zeros([3]))
    }
}
