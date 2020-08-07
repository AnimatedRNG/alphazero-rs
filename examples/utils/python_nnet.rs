use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use alphazero_rs::nnet::*;
use numpy::{IntoPyArray, PyArray1, PyArray2};
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
    last_model: usize,
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
            last_model: 0,
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

        let checkpoint_str = self
            .checkpoint
            .clone()
            .into_os_string()
            .into_string()
            .unwrap();

        let model = if self.last_model != previous_model_id {
            module
                .call(
                    "load_checkpoint",
                    (self.model.as_ref(py), model_id, checkpoint_str.clone()),
                    None,
                )
                .unwrap()
        } else {
            self.model.as_ref(py)
        };

        let new_model = module
            .call("train_model", (self.model.as_ref(py), examples), None)
            .unwrap();
        module
            .call("save_checkpoint", (model, model_id, checkpoint_str), None)
            .unwrap();

        self.model = new_model.to_object(py);
        self.last_model = model_id;
    }

    fn predict(
        &self,
        board: BatchedBoardFeaturesView,
        model_id: usize,
    ) -> (BatchedPolicy, BatchedValue) {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let board: PyObject = board.to_owned().into_pyarray(py).into();

        let checkpoint_str = self
            .checkpoint
            .clone()
            .into_os_string()
            .into_string()
            .unwrap();

        let module = self.module.as_ref(py);
        let model = if self.last_model != model_id {
            module
                .call(
                    "load_checkpoint",
                    (self.model.as_ref(py), model_id, checkpoint_str),
                    None,
                )
                .unwrap()
        } else {
            self.model.as_ref(py)
        };

        let predictions: &PyAny = module.call("predict_model", (model, board), None).unwrap();
        let predictions = predictions.downcast::<PyTuple>().unwrap();
        let (policy, value): (&PyArray2<f32>, &PyArray1<f32>) = (
            predictions.get_item(0).extract().unwrap(),
            predictions.get_item(1).extract().unwrap(),
        );

        let (policy, value) = (policy.readonly(), value.readonly());

        (
            policy.as_array().into_owned(),
            value.as_array().into_owned(),
        )
    }
}
