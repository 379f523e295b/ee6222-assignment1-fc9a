'''ee6222har dataset.'''
import os
__filedir__ = os.path.abspath(os.path.dirname(__file__))
import pathlib
import functools
import typing

import pandas as pd
import tensorflow_datasets as tfds


class BaseArtifact:
    @property
    def data(self):
        raise NotImplementedError


class CommonArtifact(BaseArtifact):
    def __init__(self, path_or_buf, **pd_kwds):
        self.path_or_buf = path_or_buf
        self.pd_kwds = pd_kwds

    @property
    def path(self) -> pathlib.Path:
        if not isinstance(self.path_or_buf, (str, os.PathLike)):
            return None
        return pathlib.Path(self.path_or_buf)


class LabelArtifact(CommonArtifact, BaseArtifact):
    class Accessor:
        @classmethod
        def read(cls, p, **pd_kwds):
            return pd.read_table(
                p,
                header=None,
                names=['class_id', 'class_name'],
                **pd_kwds
            )
    
    @functools.cached_property
    def data(self):
        return self.Accessor.read(
            self.path_or_buf,
            **self.pd_kwds
        )


class LabeledVideoArtifact(CommonArtifact, BaseArtifact):
    class Accessor:
        @classmethod
        def read(cls, p, **pd_kwds):
            return pd.read_table(
                p,
                header=None,
                names=['vid_id', 'class_id', 'vid_path'],
                **pd_kwds
            )
    
    @functools.cached_property
    def data(self):
        df = self.Accessor.read(
            self.path_or_buf,
            **self.pd_kwds
        )
        
        # TODO NOTE path conventions!!!!!!! <name>.txt, <name>
        df['vid_path'] = df['vid_path'].apply(
            lambda p: (
                self.path.parent / self.path.stem / p
            )
        )
        
        return df


class ArtifactCollection(typing.NamedTuple):
    video: typing.Any
    label: typing.Any  
    
    @classmethod
    def from_dir(cls, dirpath: str | os.PathLike):
        dirpath = pathlib.Path(dirpath)
        return cls(
            video={
                'train': LabeledVideoArtifact(dirpath / 'train.txt'),
                'val': LabeledVideoArtifact(dirpath / 'validate.txt'),
            },
            label=LabelArtifact(dirpath / 'mapping_table_23.txt'),
        )



from . import features

class Builder(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('0.0.0')
    RELEASE_NOTES = {
        '0.0.0': 'Initial release',
    }

    # TODO NOTE url: path to the zip file!!!!!!!!!!!!!
    def __init__(
        self, 
        url_or_path: str | os.PathLike = None,
        n_channels: int = 3,
        **kwargs
    ):
        self._url_or_path = (
            url_or_path or
            pathlib.Path(__filedir__) / 'data' / 'ee6222har_data.zip'
        )
        self._n_channels = n_channels
        super().__init__(**kwargs)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description='EE6222 HAR Dataset',
            supervised_keys=tuple(['frameseq', 'label']),
        )

    def _get_artifacts(
        self,
        dl_manager: tfds.download.DownloadManager,
    ) -> ArtifactCollection:
        return ArtifactCollection.from_dir(
            dl_manager.download_and_extract(
                str(self._url_or_path)
            )            
        )

    def _download_and_prepare(
        self,
        dl_manager: tfds.download.DownloadManager,
        download_config: tfds.download.DownloadConfig,
    ) -> None:
        artifacts = self._get_artifacts(dl_manager)

        self.info._features = tfds.features.FeaturesDict({
            'frameseq': tfds.features.Video(
                shape=(
                    None, None, None, 
                    self._n_channels
                ),
            ),
            'label': features.ClassLabel(
                mapping=(
                    artifacts.label.data
                    .set_index('class_id')['class_name']
                    .to_dict()
                ),
                dtype=artifacts.label.data['class_id'].dtype
            ),
        })

        return super()._download_and_prepare(dl_manager, download_config)

    def _split_generators(self, dl_manager):
        artifacts = self._get_artifacts(dl_manager)

        return {
            key: (
                (d['vid_id'], {
                    'frameseq': d['vid_path'], 
                    'label': d['class_id'],
                }) 
                for _, d in artifacts.video[a_key].data.iterrows()
            ) for key, a_key in [
                ('train', 'train'),
                ('val', 'val'),
            ]
        }

    def _generate_examples(self, _data_path):
        raise NotImplementedError
    

__all__ = [
    Builder
]
