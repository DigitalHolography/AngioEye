import h5py

from ..base import ProcessPipeline, ProcessResult, registerPipeline


@registerPipeline(name="WomersleyModeling")
class WomersleyModeling(ProcessPipeline):
    description = "TODO."

    v_profile = "/Artery/CrossSections/VelocityProfilesSegInterpOneBeat/value"

    def run(self, h5file: h5py.File) -> ProcessResult:
        """
        Executes the Womersley Modeling pipeline.
        """

        obj = h5file[self.v_profile]

        if not isinstance(obj, h5py.Dataset):
            raise ValueError(
                f"The path '{self.v_profile}' does not point to a valid dataset in the HDF5 file."
            )
        v_profile = obj[:]

        print(f"=== Data Structure for '{self.v_profile}' ===")
        print(f"HDF5 Object Type: {type(obj)}")
        print(f"Data Shape: {v_profile.shape}")
        print(f"Data Dtype: {v_profile.dtype}")

        if obj.attrs:
            print("Attributes:")
            for key, val in obj.attrs.items():
                print(f"  - {key}: {val}")
        print("=================================================================")

        return ProcessResult(metrics={}, attrs={})
