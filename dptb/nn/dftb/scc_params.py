import copy
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch

from dptb.data.transforms import OrbitalMapper
from dptb.nn.dftb.sk_param import SKParam
from dptb.nn.sktb.HubbardUDB import Hubbard_U_dict
from dptb.nn.sktb.electronic_configDB import electronic_config_dict
from dptb.nn.sktb.massDB import mass_dict
from dptb.nn.sktb.onsiteDB import onsite_energy_database
from dptb.utils.constants import Harte2eV

log = logging.getLogger(__name__)


@dataclass
class SCCParams:
    """Container for SCC-only physical parameters.

    The ``skdict`` here is intentionally not the full SK parameter dictionary
    used by ``SKParam``. It only stores the SCC physical-parameter tensors:
    ``HubdU``, ``Occu``, ``Mass``, and ``Atom_U``.
    """

    idp_sk: OrbitalMapper
    basis: Dict[str, Union[str, list]]
    skdict: Dict[str, torch.Tensor]
    r_max: Optional[Dict[str, float]] = None
    r_min: Optional[Dict[str, float]] = None
    bond_r_max: Optional[Dict[str, float]] = None
    bond_r_min: Optional[Dict[str, float]] = None
    repulsive: Optional[dict] = None

    @classmethod
    def from_skparam(cls, skp: SKParam, hubbard_u_mode: str = "dftb_atom") -> "SCCParams":
        r_max = None
        if skp.bond_r_max is not None:
            r_max = {}
            for el in skp.idp_sk.basis.keys():
                r_max[el] = max(v for k, v in skp.bond_r_max.items() if el in k.split('-'))

        skdict = {key: skp.skdict[key] for key in ["HubdU", "Occu", "Mass"]}
        if hubbard_u_mode == "dftb_atom":
            skdict["Atom_U"] = _dftb_atom_hubbard_u(skdict["HubdU"])
        elif hubbard_u_mode == "highest_occupied":
            skdict["Atom_U"] = skp.skdict["Highest_Occu_U"].clone()
        else:
            raise ValueError("hubbard_u_mode must be either 'dftb_atom' or 'highest_occupied'.")

        return cls(
            idp_sk=skp.idp_sk,
            basis=skp.idp_sk.basis,
            skdict=skdict,
            r_max=r_max,
            bond_r_max=skp.bond_r_max,
            bond_r_min=skp.bond_r_min,
        )

    @classmethod
    def from_model(cls, model) -> Optional["SCCParams"]:
        model_options = getattr(model, "model_options", None) or {}
        metadata = model_options.get("scc")
        if metadata is None:
            metadata = getattr(model, "scc_metadata", None)
        if metadata is None:
            return None
        return cls.from_options(
            basis=model.basis,
            idp_sk=model.idp_sk,
            options={"use_database": False},
            model=model,
            dtype=getattr(model, "dtype", torch.float64),
            device=getattr(model, "device", torch.device("cpu")),
        )

    @classmethod
    def from_database(
        cls,
        basis: Dict[str, Union[str, list]],
        idp_sk: OrbitalMapper,
        dtype: Union[str, torch.dtype] = torch.float64,
        device: Union[str, torch.device] = torch.device("cpu"),
        r_max: Optional[Dict[str, float]] = None,
    ) -> "SCCParams":
        return cls.from_options(
            basis=basis,
            idp_sk=idp_sk,
            options={"use_database": True, "r_max": r_max},
            dtype=dtype,
            device=device,
        )

    @classmethod
    def from_options(
        cls,
        basis: Dict[str, Union[str, list]],
        idp_sk: OrbitalMapper,
        options: Optional[dict] = None,
        model=None,
        dtype: Union[str, torch.dtype] = torch.float64,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> "SCCParams":
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        options = copy.deepcopy(options) if options is not None else {}
        use_database = options.get("use_database", True)
        hubbard_u_mode = options.get("hubbard_u_mode", "dftb_atom")
        if hubbard_u_mode not in {"dftb_atom", "highest_occupied"}:
            raise ValueError("hubbard_u_mode must be either 'dftb_atom' or 'highest_occupied'.")
        metadata = None
        if model is not None:
            model_options = getattr(model, "model_options", None) or {}
            metadata = copy.deepcopy(model_options.get("scc"))
            if metadata is None:
                metadata = copy.deepcopy(getattr(model, "scc_metadata", None))

        hubbard_u = torch.zeros([idp_sk.num_types, idp_sk.n_onsite_Es, 1], dtype=dtype, device=device)
        occupation = torch.zeros_like(hubbard_u)
        mass = torch.zeros([idp_sk.num_types, 1], dtype=dtype, device=device)

        missing_hubbard = []
        missing_occupation = []
        missing_mass = []

        explicit_hubbard = options.get("hubbard_u", {}) or {}
        explicit_occupation = options.get("occupation", {}) or {}
        explicit_mass = options.get("mass", {}) or {}
        explicit_onsite_e = options.get("onsite_e", {}) or {}
        explicit_atom_u = options.get("atom_hubbard_u", options.get("highest_occu_u", {})) or {}
        metadata_hubbard = (metadata or {}).get("hubbard_u", {}) or {}
        metadata_occupation = (metadata or {}).get("occupation", {}) or {}
        metadata_mass = (metadata or {}).get("mass", {}) or {}
        metadata_atom_u = (metadata or {}).get("atom_hubbard_u", (metadata or {}).get("highest_occu_u", {})) or {}

        for symbol, type_idx in idp_sk.chemical_symbol_to_type.items():
            for orb in idp_sk.basis[symbol]:
                full_orb = idp_sk.basis_to_full_basis[symbol][orb]
                onsite_idx = idp_sk.skonsite_maps[f"{full_orb}-{full_orb}"]

                input_orb = idp_sk.full_basis_to_basis[symbol].get(full_orb, full_orb)
                hub_value, hub_source = _resolve_orbital_value(
                    symbol=symbol,
                    orbital=full_orb,
                    input_orbital=input_orb,
                    explicit=explicit_hubbard,
                    metadata=metadata_hubbard,
                    database=Hubbard_U_dict,
                    use_database=use_database,
                    database_factor=Harte2eV,
                )
                if hub_value is None:
                    missing_hubbard.append(f"{symbol}.{full_orb}")
                else:
                    hubbard_u[type_idx, onsite_idx, 0] = hub_value
                    if hub_source == "database":
                        log.info(f"Using database Hubbard U for {symbol} {full_orb}.")

                occ_value, occ_source = _resolve_orbital_value(
                    symbol=symbol,
                    orbital=full_orb,
                    input_orbital=input_orb,
                    explicit=explicit_occupation,
                    metadata=metadata_occupation,
                    database=_occupation_database(),
                    use_database=use_database,
                )
                if occ_value is None:
                    missing_occupation.append(f"{symbol}.{full_orb}")
                else:
                    occupation[type_idx, onsite_idx, 0] = occ_value
                    if occ_source == "database":
                        log.info(f"Using database occupation for {symbol} {full_orb}.")

            mass_value, mass_source = _resolve_element_value(
                symbol=symbol,
                explicit=explicit_mass,
                metadata=metadata_mass,
                database=mass_dict,
                use_database=use_database,
            )
            if mass_value is None:
                missing_mass.append(symbol)
            else:
                mass[type_idx, 0] = mass_value
                if mass_source == "database":
                    log.info(f"Using database atomic mass for {symbol}.")

        missing = []
        if missing_hubbard:
            missing.append("hubbard_u: " + ", ".join(missing_hubbard))
        if missing_occupation:
            missing.append("occupation: " + ", ".join(missing_occupation))
        if missing_mass:
            missing.append("mass: " + ", ".join(missing_mass))
        if missing:
            raise ValueError("Missing required SCC parameters (" + "; ".join(missing) + ").")

        atom_u = _resolve_atom_u(
            idp_sk=idp_sk,
            explicit=explicit_atom_u,
            metadata=metadata_atom_u,
            dtype=dtype,
            device=device,
        )
        if atom_u is None:
            if hubbard_u_mode == "dftb_atom":
                atom_u = _dftb_atom_hubbard_u(hubbard_u)
            else:
                onsite_e = _resolve_onsite_energies(
                    idp_sk=idp_sk,
                    explicit=explicit_onsite_e,
                    metadata={},
                    dtype=dtype,
                    device=device,
                    use_database=use_database,
                )
                atom_u = _highest_occupied_hubbard_u(hubbard_u, occupation, onsite_e, dtype=dtype, device=device)
        r_max = options.get("r_max")
        if r_max is None and metadata is not None:
            r_max = metadata.get("r_max")
        if r_max is None and model is not None:
            r_max = _infer_r_max_from_model(model)
        if r_max is None:
            raise ValueError("Missing r_max for SCC graph construction. Provide scc_options['r_max'] or a model cutoff.")

        skdict = {
            "HubdU": hubbard_u,
            "Occu": occupation,
            "Mass": mass,
            "Atom_U": atom_u,
        }
        repulsive = options.get("repulsive")
        if repulsive is None and metadata is not None:
            repulsive = metadata.get("repulsive")
        return cls(idp_sk=idp_sk, basis=basis, skdict=skdict, r_max=r_max, repulsive=repulsive)

    def to_metadata(self) -> dict:
        hubbard_u = {}
        occupation = {}
        atom_u = {}
        mass = {}
        atom_u_tensor = _get_atom_u_tensor(self.skdict)
        for symbol, type_idx in self.idp_sk.chemical_symbol_to_type.items():
            hubbard_u[symbol] = {}
            occupation[symbol] = {}
            atom_u[symbol] = float(atom_u_tensor[type_idx, 0, 0].detach().cpu())
            for orb in self.idp_sk.basis[symbol]:
                full_orb = self.idp_sk.basis_to_full_basis[symbol][orb]
                onsite_idx = self.idp_sk.skonsite_maps[f"{full_orb}-{full_orb}"]
                hubbard_u[symbol][full_orb] = float(self.skdict["HubdU"][type_idx, onsite_idx, 0].detach().cpu())
                occupation[symbol][full_orb] = float(self.skdict["Occu"][type_idx, onsite_idx, 0].detach().cpu())
            mass[symbol] = float(self.skdict["Mass"][type_idx, 0].detach().cpu())
        metadata = {
            "hubbard_u": hubbard_u,
            "occupation": occupation,
            "atom_hubbard_u": atom_u,
            "mass": mass,
        }
        if self.r_max is not None:
            metadata["r_max"] = self.r_max
        if self.repulsive is not None:
            metadata["repulsive"] = self.repulsive
        return metadata


def _occupation_database() -> dict:
    return {symbol: data["valence"] for symbol, data in electronic_config_dict.items()}


def _resolve_orbital_value(symbol, orbital, explicit, metadata, database, use_database, database_factor=1.0, input_orbital=None):
    value = _nested_get(explicit, symbol, orbital)
    if value is None and input_orbital is not None:
        value = _nested_get(explicit, symbol, input_orbital)
    if value is None:
        value = _element_get(explicit, symbol)
    if value is not None:
        return float(value), "explicit"
    value = _nested_get(metadata, symbol, orbital)
    if value is None and input_orbital is not None:
        value = _nested_get(metadata, symbol, input_orbital)
    if value is None:
        value = _element_get(metadata, symbol)
    if value is not None:
        return float(value), "metadata"
    if use_database:
        value = _nested_get(database, symbol, orbital)
        if value is None and input_orbital is not None:
            value = _nested_get(database, symbol, input_orbital)
        if value is None:
            value = _element_get(database, symbol)
        if value is not None:
            return float(value) * database_factor, "database"
    return None, None


def _resolve_element_value(symbol, explicit, metadata, database, use_database):
    value = explicit.get(symbol) if isinstance(explicit, dict) else None
    if value is not None:
        return float(value), "explicit"
    value = metadata.get(symbol) if isinstance(metadata, dict) else None
    if value is not None:
        return float(value), "metadata"
    if use_database:
        value = database.get(symbol)
        if value is not None:
            return float(value), "database"
    return None, None


def _resolve_atom_u(idp_sk, explicit, metadata, dtype, device):
    if not explicit and not metadata:
        return None
    values = torch.zeros([idp_sk.num_types, 1, 1], dtype=dtype, device=device)
    missing = []
    for symbol, type_idx in idp_sk.chemical_symbol_to_type.items():
        value = _element_get(explicit, symbol)
        if value is None:
            value = _element_get(metadata, symbol)
        if value is None:
            missing.append(symbol)
        else:
            values[type_idx, 0, 0] = float(value)
    if missing:
        raise ValueError("Missing atom_hubbard_u values required for SCC metadata/options (" + ", ".join(missing) + ").")
    return values


def _nested_get(values, symbol, orbital):
    if not isinstance(values, dict):
        return None
    symbol_values = values.get(symbol)
    if not isinstance(symbol_values, dict):
        return None
    return symbol_values.get(orbital)


def _element_get(values, symbol):
    if not isinstance(values, dict):
        return None
    value = values.get(symbol)
    if isinstance(value, dict):
        return None
    return value


def _resolve_onsite_energies(idp_sk, explicit, metadata, dtype, device, use_database):
    onsite_e = torch.zeros([idp_sk.num_types, idp_sk.n_onsite_Es, 1], dtype=dtype, device=device)
    missing = []
    for symbol, type_idx in idp_sk.chemical_symbol_to_type.items():
        for orb in idp_sk.basis[symbol]:
            full_orb = idp_sk.basis_to_full_basis[symbol][orb]
            input_orb = idp_sk.full_basis_to_basis[symbol].get(full_orb, full_orb)
            onsite_idx = idp_sk.skonsite_maps[f"{full_orb}-{full_orb}"]
            value, _ = _resolve_orbital_value(
                symbol=symbol,
                orbital=full_orb,
                input_orbital=input_orb,
                explicit=explicit,
                metadata=metadata,
                database=onsite_energy_database,
                use_database=use_database,
            )
            if value is None:
                missing.append(f"{symbol}.{full_orb}")
            else:
                onsite_e[type_idx, onsite_idx, 0] = value
    if missing:
        raise ValueError("Missing onsite_e values required to identify highest-occupied atom Hubbard U (" + ", ".join(missing) + ").")
    return onsite_e


def _highest_occupied_hubbard_u(hubbard_u, occupation, onsite_e, dtype, device):
    mask = occupation > 0
    score = torch.where(mask, onsite_e, torch.tensor(float('-inf'), dtype=dtype, device=device))
    max_indices = torch.argmax(score, dim=1, keepdim=True)
    atom_u = torch.gather(hubbard_u, 1, max_indices)
    valid_rows_mask = mask.any(dim=1, keepdim=True)
    return atom_u * valid_rows_mask.to(dtype=dtype)


def _dftb_atom_hubbard_u(hubbard_u):
    """DFTB+ atom-resolved SCC uses the first shell Hubbard U for all shells."""
    return hubbard_u[:, 0:1, :].clone()


def _get_atom_u_tensor(skdict):
    if "Atom_U" in skdict:
        return skdict["Atom_U"]
    return skdict["Highest_Occu_U"]


def _infer_r_max_from_model(model):
    hopping_options = getattr(model, "hopping_options", None)
    if hopping_options is None:
        return None
    rs = hopping_options.get("rs")
    basis = getattr(model, "basis", None)
    if rs is None or basis is None:
        return None
    if isinstance(rs, (int, float)):
        return {symbol: float(rs) for symbol in basis.keys()}
    if isinstance(rs, dict):
        r_max = {}
        for symbol in basis.keys():
            values = []
            for key, value in rs.items():
                parts = key.split("-")
                if symbol in parts:
                    values.append(float(value))
            if values:
                r_max[symbol] = max(values)
        if len(r_max) == len(basis):
            return r_max
    return None
