import joblib

from src.core.config import Config
from src.core.logger import logger
from src.models.model_factory import ModelFactory



class ModelTrainer:
    """
    Handles training and saving of machine learning models.
    """

    @staticmethod
    def train_model(
        model_name,
        model,
        X_train,
        y_train,
    ):
        """
        Train a single machine learning model
        and save the trained model.
        """

        logger.info("=" * 50)
        logger.info(f"TRAINING MODEL: {model_name}")
        logger.info("=" * 50)

        model.fit(
            X_train,
            y_train,
        )

        logger.info(
            f"{model_name} trained successfully."
        )

        model_path = (
            Config.MODEL_DIR
            / f"{model_name}.pkl"
        )

        joblib.dump(
            model,
            model_path,
        )

        logger.info(
            f"Saved model to: {model_path}"
        )

        return model


    @classmethod
    def train_all_models(
        cls,
        X_train,
        y_train,
    ):
        """
        Train all models provided by ModelFactory.

        Parameters
        ----------
        X_train :
            Training feature matrix.

        y_train :
            Training target labels.

        Returns
        -------
        dict
            Dictionary containing all trained models.
        """

        logger.info("=" * 60)
        logger.info("STARTING ALL MODEL TRAINING")
        logger.info("=" * 60)

        # Get all model instances
        models = ModelFactory.get_all_models()

        trained_models = {}

        # Train each model
        for model_name, model in models.items():

            logger.info(
                f"Starting training: {model_name}"
            )

            trained_model = cls.train_model(
                model_name=model_name,
                model=model,
                X_train=X_train,
                y_train=y_train,
            )

            trained_models[model_name] = trained_model

            logger.info(
                f"Completed training: {model_name}"
            )

        logger.info("=" * 60)
        logger.info("ALL MODELS TRAINED SUCCESSFULLY")
        logger.info("=" * 60)

        logger.info(
            f"Total models trained: {len(trained_models)}"
        )

        return trained_models