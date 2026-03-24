from .mnist import MNIST_Dataset
from .fmnist import FashionMNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .odds import ODDSADDataset
from .attack_csv import AttackCSVDataset


def load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                 ratio_known_normal, ratio_known_outlier, ratio_pollution, random_state):

    implemented_datasets = (
        'mnist', 'fmnist', 'cifar10',
        'arrhythmia', 'cardio', 'satellite', 'satimage-2', 'shuttle', 'thyroid',
        'attack_csv'
    )
    assert dataset_name in implemented_datasets, f"dataset_name={dataset_name} 不在支持列表中"

    if dataset_name == 'mnist':
        from .mnist import MNIST_Dataset
        dataset = MNIST_Dataset(
            root=data_path,
            normal_class=normal_class,
            known_outlier_class=known_outlier_class,
            n_known_outlier_classes=n_known_outlier_classes,
            ratio_known_normal=ratio_known_normal,
            ratio_known_outlier=ratio_known_outlier,
            ratio_pollution=ratio_pollution,
            random_state=random_state
        )

    elif dataset_name == 'fmnist':
        from .fmnist import FashionMNIST_Dataset
        dataset = FashionMNIST_Dataset(
            root=data_path,
            normal_class=normal_class,
            known_outlier_class=known_outlier_class,
            n_known_outlier_classes=n_known_outlier_classes,
            ratio_known_normal=ratio_known_normal,
            ratio_known_outlier=ratio_known_outlier,
            ratio_pollution=ratio_pollution,
            random_state=random_state
        )

    elif dataset_name == 'cifar10':
        from .cifar10 import CIFAR10_Dataset
        dataset = CIFAR10_Dataset(
            root=data_path,
            normal_class=normal_class,
            known_outlier_class=known_outlier_class,
            n_known_outlier_classes=n_known_outlier_classes,
            ratio_known_normal=ratio_known_normal,
            ratio_known_outlier=ratio_known_outlier,
            ratio_pollution=ratio_pollution,
            random_state=random_state
        )

    elif dataset_name in ('arrhythmia', 'cardio', 'satellite', 'satimage-2', 'shuttle', 'thyroid'):
        from .odds import ODDSADDataset
        dataset = ODDSADDataset(
            root=data_path,
            dataset_name=dataset_name,
            n_known_outlier_classes=n_known_outlier_classes,
            ratio_known_normal=ratio_known_normal,
            ratio_known_outlier=ratio_known_outlier,
            ratio_pollution=ratio_pollution,
            random_state=random_state
        )

    elif dataset_name == 'attack_csv':
        from .attack_csv import AttackCSVDataset
        dataset = AttackCSVDataset(
            root=data_path,
            normal_class=normal_class,
            known_outlier_class=known_outlier_class,
            ratio_known_normal=ratio_known_normal,
            ratio_known_outlier=ratio_known_outlier,
            ratio_pollution=ratio_pollution
        )

    else:
        raise ValueError(f"未知数据集: {dataset_name}")

    return dataset