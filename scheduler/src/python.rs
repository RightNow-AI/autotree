use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAny, PyAnyMethods, PyDict, PyDictMethods, PyList, PyListMethods, PyModule},
};

use crate::{
    BeamConfig, BestFirstConfig, BranchId, Command, EngineEvent, MctsConfig, PolicyConfig,
    Scheduler as CoreScheduler, SchedulerConfig, SchedulerError,
};

#[pyclass(name = "Scheduler")]
pub struct PyScheduler {
    inner: CoreScheduler,
}

#[pymethods]
impl PyScheduler {
    #[new]
    fn new(config: &Bound<'_, PyDict>) -> PyResult<Self> {
        let policy_name: String = required_item(config, "policy")?.extract()?;
        let branches = optional_u32(config, "branches", 4)?;
        let max_depth = optional_u32(config, "max_depth", 8)?;
        let policy = match policy_name.as_str() {
            "beam" => PolicyConfig::Beam(BeamConfig {
                width: branches,
                fork_width: optional_u32(config, "fork_width", branches)?,
                fork_at_tokens: optional_u64_vec(config, "fork_at_tokens", vec![1])?,
            }),
            "best-first" | "best_first" => PolicyConfig::BestFirst(BestFirstConfig {
                expansion_width: branches,
                max_depth,
            }),
            "mcts" => PolicyConfig::Mcts(MctsConfig {
                expansion_width: branches,
                max_depth,
                exploration_weight: optional_f64(
                    config,
                    "exploration_weight",
                    std::f64::consts::SQRT_2,
                )?,
            }),
            _ => {
                return Err(PyValueError::new_err(
                    "policy must be 'beam', 'best-first', or 'mcts'",
                ));
            }
        };

        let total_token_budget = optional_u64(config, "budget_tokens", 4_000)?;
        let scheduler_config = SchedulerConfig {
            policy,
            seed: optional_u64(config, "seed", 0)?,
            total_token_budget,
            per_branch_token_budget: optional_u64(
                config,
                "per_branch_token_budget",
                total_token_budget,
            )?,
            speculative_kill_margin: optional_nullable_f64(config, "speculative_kill_margin")?,
        };
        let scorer = optional_string(config, "scorer", "logprob")?;
        let inner = match scorer.as_str() {
            "logprob" => CoreScheduler::new(scheduler_config),
            "external" | "value_head" => CoreScheduler::with_external_values(scheduler_config),
            _ => {
                return Err(PyValueError::new_err(
                    "scorer must be 'logprob', 'external', or 'value_head'",
                ));
            }
        }
        .map_err(to_python_error)?;
        Ok(Self { inner })
    }

    fn feed_event(&mut self, event: &Bound<'_, PyDict>) -> PyResult<()> {
        let event_type: String = required_item(event, "type")?.extract()?;
        let branch = BranchId(required_item(event, "branch")?.extract()?);
        let parsed = match event_type.as_str() {
            "token_sampled" => EngineEvent::TokenSampled {
                branch,
                token: required_item(event, "token")?.extract()?,
                logprob: required_item(event, "logprob")?.extract()?,
            },
            "branch_exhausted" => EngineEvent::BranchExhausted { branch },
            "value_scored" => EngineEvent::ValueScored {
                branch,
                score: required_item(event, "score")?.extract()?,
            },
            _ => {
                return Err(PyValueError::new_err(
                    "event type must be 'token_sampled', 'branch_exhausted', or 'value_scored'",
                ));
            }
        };
        self.inner.feed_event(parsed).map_err(to_python_error)
    }

    fn poll_commands(&mut self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let output = PyList::empty(py);
        for command in self.inner.poll_commands() {
            let item = PyDict::new(py);
            match command {
                Command::ForkAt { branch, width } => {
                    item.set_item("type", "fork_at")?;
                    item.set_item("branch", branch.0)?;
                    item.set_item("width", width)?;
                }
                Command::Kill { branch, reason } => {
                    item.set_item("type", "kill")?;
                    item.set_item("branch", branch.0)?;
                    item.set_item("reason", reason.as_str())?;
                }
                Command::Continue { branch } => {
                    item.set_item("type", "continue")?;
                    item.set_item("branch", branch.0)?;
                }
                Command::Finalize { branch } => {
                    item.set_item("type", "finalize")?;
                    item.set_item("branch", branch.0)?;
                }
            }
            output.append(item)?;
        }
        Ok(output.unbind())
    }

    fn drain(&mut self) -> PyResult<()> {
        self.inner.drain().map_err(to_python_error)
    }
}

fn required_item<'py>(dictionary: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    dictionary
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing required field '{key}'")))
}

fn optional_u64(dictionary: &Bound<'_, PyDict>, key: &str, default: u64) -> PyResult<u64> {
    dictionary
        .get_item(key)?
        .map_or(Ok(default), |value| value.extract())
}

fn optional_u32(dictionary: &Bound<'_, PyDict>, key: &str, default: u32) -> PyResult<u32> {
    dictionary
        .get_item(key)?
        .map_or(Ok(default), |value| value.extract())
}

fn optional_f64(dictionary: &Bound<'_, PyDict>, key: &str, default: f64) -> PyResult<f64> {
    dictionary
        .get_item(key)?
        .map_or(Ok(default), |value| value.extract())
}

fn optional_string(dictionary: &Bound<'_, PyDict>, key: &str, default: &str) -> PyResult<String> {
    dictionary
        .get_item(key)?
        .map_or_else(|| Ok(default.to_owned()), |value| value.extract())
}

fn optional_nullable_f64(dictionary: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<f64>> {
    match dictionary.get_item(key)? {
        Some(value) if !value.is_none() => value.extract().map(Some),
        Some(_) | None => Ok(None),
    }
}

fn optional_u64_vec(
    dictionary: &Bound<'_, PyDict>,
    key: &str,
    default: Vec<u64>,
) -> PyResult<Vec<u64>> {
    dictionary
        .get_item(key)?
        .map_or(Ok(default), |value| value.extract())
}

fn to_python_error(error: SchedulerError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pymodule]
fn autotree_scheduler(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyScheduler>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_surface_round_trips_events_commands_and_drain() {
        Python::initialize();
        Python::attach(|py| {
            let config = PyDict::new(py);
            config.set_item("policy", "beam").unwrap();
            config.set_item("branches", 2).unwrap();
            config.set_item("fork_width", 2).unwrap();
            config.set_item("fork_at_tokens", vec![1_u64]).unwrap();
            config.set_item("budget_tokens", 100).unwrap();
            let mut scheduler = PyScheduler::new(&config).unwrap();

            let event = PyDict::new(py);
            event.set_item("type", "token_sampled").unwrap();
            event.set_item("branch", 0).unwrap();
            event.set_item("token", 7).unwrap();
            event.set_item("logprob", -0.1).unwrap();
            scheduler.feed_event(&event).unwrap();
            assert_eq!(scheduler.poll_commands(py).unwrap().bind(py).len(), 3);

            scheduler.drain().unwrap();
            assert_eq!(scheduler.poll_commands(py).unwrap().bind(py).len(), 3);
        });
    }
}
