
from tqdm import tqdm
import os
import torch.nn as nn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import sys
import importlib

from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from .LLMManager import EmbeddingManager
# from ..config.llm_config import cfg
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb


class ClassifierManager:
    def __init__(self, classifier_type='linear', require_grad: bool=False):
        self.type = classifier_type
        self.classifiers = []
        self.testacc = []
        self.require_grad = require_grad
        

    def _train_classifiers(
        self, 
        pos_embds: EmbeddingManager,
        neg_embds: EmbeddingManager,
        lr: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 32,
    ):
        print("Training classifiers...")

        self.llm_cfg = pos_embds.llm_cfg

        for i in tqdm(range(self.llm_cfg.n_layer)):
            if self.type == "linear":
                layer_classifier = LinearLayerClassifier(pos_embds.llm_cfg, lr)
            elif self.type == "mlp":
                layer_classifier = MLPLayerClassifier(pos_embds.llm_cfg, lr)
            
            layer_classifier.train(
                pos_tensor=pos_embds.activations[i],
                neg_tensor=neg_embds.activations[i],
                n_epoch=n_epochs,
                batch_size=batch_size,
            )

            self.classifiers.append(layer_classifier)

    def _evaluate_testacc(self, pos_embds: EmbeddingManager, neg_embds: EmbeddingManager):
        for i in tqdm(range(len(self.classifiers))):
            self.testacc.append(
                self.classifiers[i].evaluate_testacc(
                    pos_tensor=pos_embds.activations[i],
                    neg_tensor=neg_embds.activations[i],
                )
            )
    
    def fit(
        self, 
        pos_embds_train: EmbeddingManager,
        neg_embds_train: EmbeddingManager,
        pos_embds_test: EmbeddingManager,
        neg_embds_test: EmbeddingManager,
        lr: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 32,
    ):
        self._train_classifiers(
            pos_embds_train,
            neg_embds_train,
            lr,
            n_epochs,
            batch_size,
        )

        self._evaluate_testacc(
            pos_embds_test,
            neg_embds_test,
        )

        return self
    
    def save(self, save_path: str):
        file_name = f"{self.type}_{self.llm_cfg.model_nickname}.pth"
        torch.save(self, os.path.join(save_path, file_name))
    
    def load(self, file_path: str):
        
        sys.modules["model"] = importlib.import_module("src.model")
        sys.modules["model.llm_config"] = importlib.import_module("src.config.llm_config")
        classifier_manager = torch.load(file_path, weights_only=False)#, map_location=torch.device("cuda:1"))
        self.__dict__.update(classifier_manager.__dict__)

        if self.require_grad:
            for classifier in self.classifiers:
                try: classifier.model.requires_grad = True
                except:
                    try: 
                        classifier.mlp.requires_grad = True
                        classifier.model = classifier.mlp  
                    except: 
                        try: 
                            classifier.linear.requires_grad = True
                            classifier.model = classifier.linear 
                        except: assert False, "No model found"

    
def load_classifier_manager(file_path: str) -> ClassifierManager:
    sys.modules["model"] = importlib.import_module("src.model")
    sys.modules["model.llm_config"] = importlib.import_module("src.config.llm_config")
    return torch.load(file_path, weights_only=False)


class LinearLayerClassifier:
    def __init__(self, llm_cfg: cfg, lr: float=0.01, max_iter: int=10000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LogisticRegression(solver="saga", max_iter=max_iter)
        
        self.data = {
            "train": {
                "pos": None,
                "neg": None,
            },
            "test": {
                "pos": None,
                "neg": None,
            }
        }

    def train(self, pos_tensor: torch.tensor, neg_tensor: torch.tensor, n_epoch: int=100, batch_size: int=64) -> list[float]:
        X = torch.vstack([pos_tensor, neg_tensor]).to(self.device)
        y = torch.cat((torch.ones(pos_tensor.size(0)), torch.zeros(neg_tensor.size(0)))).to(self.device)
        
        self.data["train"]["pos"] = pos_tensor.cpu()
        self.data["train"]["neg"] = neg_tensor.cpu()

        self.model.fit(X.cpu().numpy(), y.cpu().numpy())

        return []
    
    def predict(self, tensor: torch.tensor, require_grad: bool=False) -> torch.tensor:
        if tensor.requires_grad:
            return torch.tensor(self.model.predict(tensor), requires_grad=True)
        else:
            return torch.tensor(self.model.predict(tensor))
        
    def evaluate_testacc(self, pos_tensor: torch.tensor, neg_tensor: torch.tensor) -> float:
        test_data = torch.vstack([pos_tensor, neg_tensor]).to(self.device)
        predictions = self.predict(test_data)
        true_labels = torch.cat((torch.ones(pos_tensor.size(0)), torch.zeros(neg_tensor.size(0))))

        correct_count = torch.sum((predictions > 0.5) == true_labels).item()

        self.data["test"]["pos"] = pos_tensor.cpu()
        self.data["test"]["neg"] = neg_tensor.cpu()

        return correct_count / len(true_labels)
    
    def get_weights_bias(self) -> tuple[torch.tensor, torch.tensor]:
        return torch.tensor(self.model.coef_).to(self.device), torch.tensor(self.model.intercept_).to(self.device)

class MLPLayerClassifier:
    def __init__(self, llm_cfg: cfg, lr: float=0.01, max_iter: int=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=max_iter)
        self.model = MLPClassifierTorch(max_iter=max_iter, device=self.device, input_size=llm_cfg.n_dimension)
        
        self.data = {
            "train": {
                "pos": None,
                "neg": None,
            },
            "test": {
                "pos": None,
                "neg": None,
            }
        }

    def train(self, pos_tensor: torch.tensor, neg_tensor: torch.tensor, n_epoch: int=100, batch_size: int=64) -> list[float]:
        X = torch.vstack([pos_tensor, neg_tensor]).to(self.device)
        y = torch.cat((torch.ones(pos_tensor.size(0)), torch.zeros(neg_tensor.size(0)))).to(self.device)
        
        self.data["train"]["pos"] = pos_tensor.cpu()
        self.data["train"]["neg"] = neg_tensor.cpu()

        self.model.fit(X.cpu().numpy(), y.cpu().numpy())

        return []
    
    def predict(self, tensor: torch.tensor) -> torch.tensor:
        t = self.model.predict(tensor)
        return torch.softmax(t, dim=1)
        return self.model.predict(tensor)

    def predict_logits(self, tensor: torch.tensor) -> torch.tensor:
        return self.model.predict(tensor)
        
    def evaluate_testacc(self, pos_tensor: torch.tensor, neg_tensor: torch.tensor) -> float:
        test_data = torch.vstack([pos_tensor, neg_tensor]).to(self.device)
        predictions = self.predict(test_data).cpu() # shape: (batch_size, 2)
        # predictions = torch.where(predictions > 0.5, 1, 0).squeeze()
        predictions = torch.argmax(predictions, dim=1).type(torch.float32)  # shape: (batch_size,)
        true_labels = torch.cat((torch.ones(pos_tensor.size(0)), torch.zeros(neg_tensor.size(0)))).cpu()

        # print(predictions.shape)
        # print(true_labels.shape)
        correct_count = torch.sum(predictions == true_labels).item()
        # print(correct_count)
        # print(len(true_labels))

        self.data["test"]["pos"] = pos_tensor.cpu()
        self.data["test"]["neg"] = neg_tensor.cpu()

        return round(correct_count / len(true_labels), 4)

    def get_pred(self, inp_tensor) -> float:
        test_data = inp_tensor.to(self.device)
        predictions = self.predict(test_data)
        # true_labels = torch.cat((torch.ones(pos_tensor.size(0)), torch.zeros(neg_tensor.size(0))))

        pred = torch.sum(predictions).item()
        return predictions, pred / len(test_data)
    
    def get_weights_bias(self) -> tuple[torch.tensor, torch.tensor]:
        return torch.tensor(self.model.coefs_).to(self.device), torch.tensor(self.model.intercepts_).to(self.device)


class MLPClassifierTorch(nn.Module):
    def __init__(self, input_size: int=4096, hidden_size1: int=100, hidden_size2: int=50, output_size: int=2,
                 max_iter: int=200, device: str="cuda"):
        super(MLPClassifierTorch, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1, device=device)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2, device=device)
        self.fc3 = nn.Linear(hidden_size2, output_size, device=device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.max_iter = max_iter
        self.device = device
        
    def forward(self, x):
        x = x.to(self.device)
        # print("x.requires_grad 1:", x.requires_grad)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # print("x.requires_grad 2:", x.requires_grad)
        return x
    
    def predict(self, x):
        x = x.to(torch.float32)
        return self.forward(x)
    
    def fit(self, X, y):
        loss_fn = torch.nn.CrossEntropyLoss()

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        for _ in range(self.max_iter):
            self.optimizer.zero_grad()
            logits = self.forward(X)    # shape: (batch_size, num_classes=2)
            predictions = torch.softmax(logits, dim=1) 
            predictions = torch.argmax(predictions, dim=1).type(torch.float32)  # shape: (batch_size,)
            # print(logits.shape, y.shape)
            # print(logits.dtype, y.dtype)
            # print(logits.requires_grad, y.requires_grad)
            loss = loss_fn(logits, y.long())
            # print(loss.item())
            loss.backward()
            self.optimizer.step()
    
    
    
if __name__ == '__main__':
    
    
    pass