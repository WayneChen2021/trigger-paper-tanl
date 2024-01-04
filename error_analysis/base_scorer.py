import argparse
import json
import os
import matplotlib.pyplot as plt

from utils import error_analysis_event


def main(config_file):
    with open(config_file, "r") as f:
        configs = json.loads(f.read())

    muc_roles = ["PerpInd", "PerpOrg", "Target", "Victim", "Weapon"]
    wikievent_roles = [
        "Defeated",
        "Investigator",
        "Killer",
        "Jailer",
        "Destroyer",
        "ManufacturerAssembler",
        "Instrument",
        "PlaceOfEmployment",
        "Learner",
        "Components",
        "Vehicle",
        "Disabler",
        "Injurer",
        "Communicator",
        "BodyPart",
        "Disease",
        "Detainee",
        "Position",
        "Patient",
        "PassengerArtifact",
        "Defendant",
        "Attacker",
        "IdentifiedRole",
        "Preventer",
        "CrashObject",
        "Victim",
        "Identifier",
        "AcquiredEntity",
        "Researcher",
        "Regulator",
        "Observer",
        "Artifact",
        "Target",
        "Participant",
        "Recipient",
        "Topic",
        "Impeder",
        "Treater",
        "Subject",
        "Destination",
        "Giver",
        "Perpetrator",
        "ExplosiveDevice",
        "Transporter",
        "Employee",
        "PaymentBarter",
        "Place",
        "Damager",
        "JudgeCourt",
        "ArtifactMoney",
        "IdentifiedObject",
        "ObservedEntity",
        "Demonstrator",
        "Victor",
        "TeacherTrainer",
        "Prosecutor",
        "Dismantler",
        "DamagerDestroyer",
        "Origin",
    ]
    for config in configs:
        pred_dev_dir = os.path.join(config["experiment_dir"], "dev_predictions")
        pred_test_dir = os.path.join(config["experiment_dir"], "test_predictions")
        if config["dataset_name"] == "muc":
            roles = muc_roles
        elif config["dataset_name"] == "wikievents":
            roles = wikievent_roles

        if not os.path.exists(config["output_dir"]):
            os.mkdir(config["output_dir"])

        dev_f1s, dev_recalls, dev_precisions = [], [], []
        test_f1s, test_recalls, test_precisions = [], [], []

        for file_name in os.listdir(pred_dev_dir):
            epoch_name = file_name[:-6]
            iteration_dir = os.path.join(config["output_dir"], f"outputs_{epoch_name}")
            if not os.path.exists(iteration_dir):
                os.mkdir(iteration_dir)

            dev_output_path = os.path.join(iteration_dir, "dev.json")
            test_output_path = os.path.join(iteration_dir, "test.json")

            error_analysis_event(
                os.path.join(pred_dev_dir, file_name),
                config["tanl_ref_dev"],
                config["gtt_ref_dev"],
                "hungarian_scorer.py",
                config["types_mapping"],
                dev_output_path,
                roles,
                config["hungarian_config"],
                config["relax_match"],
            )

            error_analysis_event(
                os.path.join(pred_test_dir, file_name),
                config["tanl_ref_test"],
                config["gtt_ref_test"],
                "hungarian_scorer.py",
                config["types_mapping"],
                test_output_path,
                roles,
                config["hungarian_config"],
                config["relax_match"],
            )

            with open(dev_output_path, "r") as f:
                info = json.loads(f.read())
                iter_f1_dev = info["overall_f1"]
                iter_recall_dev = info["overall_recall"]
                iter_precision_dev = info["overall_precision"]

            with open(test_output_path, "r") as f:
                info = json.loads(f.read())
                iter_f1_test = info["overall_f1"]
                iter_recall_test = info["overall_recall"]
                iter_precision_test = info["overall_precision"]

            dev_f1s.append(iter_f1_dev)
            dev_recalls.append(iter_recall_dev)
            dev_precisions.append(iter_precision_dev)
            test_f1s.append(iter_f1_test)
            test_recalls.append(iter_recall_test)
            test_precisions.append(iter_precision_test)

        best_iter_f1 = max(enumerate(dev_f1s), key=lambda tup: tup[1])[0]
        print(
            f'{config["experiment_dir"]}:\nepoch {best_iter_f1}\ntest f1: {test_f1s[best_iter_f1]}\ntest recall: {test_recalls[best_iter_f1]}\ntest precision: {test_precisions[best_iter_f1]}\n\n'
        )

        iterations = [i for i in range(1, len(list(os.listdir(pred_dev_dir))) + 1)]
        plt.plot(iterations, dev_f1s, label="dev F1")
        plt.plot(iterations, dev_recalls, label="dev recall")
        plt.plot(iterations, dev_precisions, label="dev precision")
        plt.plot(iterations, test_f1s, label="test F1")
        plt.plot(iterations, test_recalls, label="test recall")
        plt.plot(iterations, test_precisions, label="test precision")
        plt.xlabel("epoch number")
        plt.ylabel("statistic")
        plt.title("Performance vs Epoch Number")
        plt.legend()
        plt.savefig(os.path.join(config["output_dir"], "plot.png"))
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=False, default="base_score_config.json"
    )
    """
    config format:
    [
        {
            "experiment_dir": ..., # path
            "output_dir": ..., # path
            "hungarian_config": ..., # path
            "types_mapping": ..., # path
            "gtt_ref_dev": ..., # file
            "tanl_ref_dev": ..., # file
            "gtt_ref_test": ..., # file
            "tanl_ref_test": ..., # file
            "relax_match": ..., True or False
            "dataset_name": ... # one of 'muc' or 'wikievents'
        },
        {
            "experiment_dir": ...,
            "output_dir": ...,
            "hungarian_config": ...,
            "types_mapping": ...,
            "gtt_ref_dev": ...,
            "tanl_ref_dev": ...,
            "gtt_ref_test": ...,
            "tanl_ref_test": ...,
            "relax_match": ...,
            "dataset_name": ...
        }
    ]
    """
    args = parser.parse_args()
    main(args.config)
