"""Misc utils."""
import logging
from pathlib import Path
from typing import List

from rich.logging import RichHandler
logger = logging.getLogger(__name__)


def setup_logger(log_dir: str):
    """Create log directory and logger."""
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    log_path = str(Path(log_dir) / "log.txt")
    handlers = [logging.FileHandler(log_path), RichHandler(rich_tracebacks=True)]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(module)s] [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def compute_metrics(preds: List, golds: List, task: str):
    """Compute metrics."""
    logger.info(f"start computing metrics! {task}")
    mets = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "crc": 0, "total": 0}
    for pred, label in zip(preds, golds):
        label = label.strip().lower()
        pred = pred.strip().lower()
        mets["total"] += 1
        if task in {
            #"data_imputation",
            # "entity_matching",
        }:
            crc = pred == label
        elif task in {"data_imputation"}:
            if "is" in pred:
                 startst = pred.split("is")[0]
                 endst = pred.split("is")[-1]
                 #logger.info(f"start is {startst}, end is {endst}, label is {label}")
                 if label in endst or label in startst:
                     crc = True
                 else:
                    crc = False
            else:
                crc = pred == label    
            logger.info(f"crc is {crc}")
        elif task in {"entity_matching", "schema_matching", "error_detection_spelling"}:
            logger.info("enter error_detection_spelling")
            logger.info(f"pred is {pred} , label is {label}")
            crc = pred.startswith(label)
            logger.info(f"crc is {crc}")
        elif task in {"error_detection"}: 
            pred = pred.split("\n\n")[-1]
            breakpoint()
            crc = pred.endswith(label)
           
        else:
            raise ValueError(f"Unknown task: {task}")
        # Measure equal accuracy for generation
        if crc:
            mets["crc"] += 1
        if label == "yes":
            if crc:
                mets["tp"] += 1
                logger.info("tp + 1")
            else:
                mets["fn"] += 1
                logger.info("fn + 1")
        elif label == "no":
            if crc:
                mets["tn"] += 1
                logger.info("tn + 1")
            else:
                mets["fp"] += 1
                logger.info("fp + 1")

    logger.info(mets["tp"])
    logger.info(mets["fp"])
    logger.info(mets["tn"])
    logger.info(mets["fn"])

              
    prec = mets["tp"] / max(1, (mets["tp"] + mets["fp"]))
    rec = mets["tp"] / max(1, (mets["tp"] + mets["fn"]))
    acc = mets["crc"] / mets["total"]
    f1 = 2 * prec * rec / max(1, (prec + rec))
    return prec, rec, acc, f1
