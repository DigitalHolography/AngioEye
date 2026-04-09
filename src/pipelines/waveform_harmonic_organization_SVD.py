import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


@registerPipeline(name="waveform_harmonic_organization_SVD")
class WaveformHarmonicOrganizationSVD(ProcessPipeline):
    """
    Rank-1 low-rank analysis of harmonic organization on beat-resolved segment waveforms.

    This pipeline implements the matrix/SVD and tensor/CP parts of the internship
    proposal on normalized complex harmonic coefficients c_hbkr = V_hbkr / V_1bkr,
    restricted to rank-1 approximations. https://doi.org/10.5281/zenodo.19430581

    Matrix formulation:
      - X_{j,h-1} = c_hbkr
      - relaxed row-validity rule: keep any beat-location pair with at least
        min_valid_harmonics_per_row valid higher harmonics
      - invalid harmonic entries inside X are zero-filled (option 1)
      - rank-1 SVD approximation only
      - eta_1
      - xi_bkr = sigma_1 * P_{j1}
      - beat/location aggregated mode scores
      - heterogeneity indices H_xi^(b), H_xi^(kr)

    Tensor formulation:
      - chi_{h-1,b,k,r} = c_hbkr
      - rank-1 CP approximation only
      - lambda_1_cp, z_h, u_b, v_k, w_r
      - chat1_hbkr
      - eta_1_cp

    Inputs
    ------
    - raw per-segment arterial waveforms:
        /Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value
    - raw per-segment venous waveforms:
        /Vein/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value
    - beat periods:
        /Artery/VelocityPerBeat/beatPeriodSeconds/value

    Expected segment layout
    -----------------------
    v_block[t, beat, branch, radius]
    """

    description = (
        "Rank-1 low-rank harmonic-organization analysis on per-segment beat-resolved "
        "arterial and venous waveforms: matrix construction, SVD, eta_1, xi, beat/"
        "location heterogeneity, and rank-1 tensor/CP decomposition."
    )

    # ----------------------------
    # Inputs
    # ----------------------------
    v_raw_segment_input_artery = (
        "/Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"
    )
    v_raw_segment_input_vein = (
        "/Vein/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"
    )
    T_input = "/Artery/VelocityPerBeat/beatPeriodSeconds/value"

    # ----------------------------
    # Parameters
    # ----------------------------
    eps = 1e-12
    H_MAX = 10
    higher_harmonic_rel_threshold = 5e-2

    # Relaxed matrix-row rule
    min_valid_harmonics_per_row = 2

    cp_max_iter = 100
    cp_tol = 1e-7

    # ----------------------------
    # Helpers
    # ----------------------------
    @staticmethod
    def _ensure_beat_periods(T: np.ndarray, n_beats: int) -> np.ndarray:
        T = np.asarray(T, dtype=float)

        if T.ndim == 1:
            if T.size != n_beats:
                raise ValueError(
                    f"Beat-period vector length mismatch: got {T.size}, expected {n_beats}"
                )
            return T

        if T.ndim == 2:
            if T.shape == (1, n_beats):
                return T[0]
            if T.shape == (n_beats, 1):
                return T[:, 0]
            if T.shape[0] == 1 and T.shape[1] >= n_beats:
                return T[0, :n_beats]
            if T.shape[1] == 1 and T.shape[0] >= n_beats:
                return T[:n_beats, 0]

        raise ValueError(f"Could not interpret beat periods with shape {T.shape}")

    @staticmethod
    def _complex_nan(shape: tuple[int, ...]) -> np.ndarray:
        return np.full(shape, np.nan + 1j * np.nan, dtype=np.complex128)

    @staticmethod
    def _isfinite_complex(x: np.ndarray) -> np.ndarray:
        return np.isfinite(np.real(x)) & np.isfinite(np.imag(x))

    @staticmethod
    def _count_true(mask: np.ndarray, axis=None) -> np.ndarray:
        return np.sum(np.asarray(mask, dtype=np.uint8), axis=axis, dtype=np.int32)

    def _complex_mean(self, x: np.ndarray, axis) -> np.ndarray:
        x = np.asarray(x, dtype=np.complex128)
        mask = self._isfinite_complex(x)
        sumx = np.sum(np.where(mask, x, 0.0 + 0.0j), axis=axis)
        count = np.sum(mask, axis=axis)

        if np.ndim(sumx) == 0:
            if count > 0:
                return np.complex128(sumx / count)
            return np.complex128(np.nan + 1j * np.nan)

        out = self._complex_nan(sumx.shape)
        valid = count > 0
        out[valid] = sumx[valid] / count[valid]
        return out

    @staticmethod
    def _safe_cv_abs(x: np.ndarray, eps: float) -> tuple[float, int]:
        vals = np.abs(x) if np.iscomplexobj(x) else np.asarray(x, dtype=float)
        valid = np.isfinite(vals)
        n_valid = int(np.sum(valid))
        if n_valid <= 1:
            return np.nan, n_valid

        vv = vals[valid]
        meanv = float(np.mean(vv))
        if (not np.isfinite(meanv)) or meanv <= eps:
            return np.nan, n_valid

        stdv = float(np.std(vv))
        return float(stdv / (meanv + eps)), n_valid

    @staticmethod
    def _normalize_complex_vector(x: np.ndarray, eps: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.complex128)
        x = np.where(np.isfinite(np.real(x)) & np.isfinite(np.imag(x)), x, 0.0 + 0.0j)
        nrm = float(np.sqrt(np.sum(np.abs(x) ** 2)))
        if (not np.isfinite(nrm)) or nrm <= eps:
            return x
        return x / nrm

    def _leading_left_singular_vector_or_uniform(self, M: np.ndarray) -> np.ndarray:
        M = np.asarray(M, dtype=np.complex128)
        n = int(M.shape[0])
        if n == 0:
            return self._complex_nan((0,))

        fro = float(np.sqrt(np.sum(np.abs(M) ** 2)))
        if (not np.isfinite(fro)) or fro <= self.eps:
            return np.ones((n,), dtype=np.complex128) / np.sqrt(float(n))

        U, _, _ = np.linalg.svd(M, full_matrices=False)
        if U.shape[1] == 0:
            return np.ones((n,), dtype=np.complex128) / np.sqrt(float(n))
        return self._normalize_complex_vector(U[:, 0], self.eps)

    # ----------------------------
    # Harmonics
    # ----------------------------
    def _harmonic_coefficients_block(
        self, v_block: np.ndarray, T_vec: np.ndarray
    ) -> dict:
        if v_block.ndim != 4:
            raise ValueError(
                f"Expected segment block shape (n_t,n_beats,n_branches,n_radii), "
                f"got {v_block.shape}"
            )

        v_block = np.asarray(v_block, dtype=float)
        n_t, n_beats, n_branches, n_radii = v_block.shape
        T_vec = self._ensure_beat_periods(T_vec, n_beats)

        H = int(min(self.H_MAX, n_t // 2))
        if H < 1:
            raise ValueError(f"Need at least one harmonic, got H={H} for n_t={n_t}")

        V_hbkr = self._complex_nan((H + 1, n_beats, n_branches, n_radii))
        valid_waveform_bkr_mask = np.zeros((n_beats, n_branches, n_radii), dtype=bool)

        for b in range(n_beats):
            Tbeat = float(T_vec[b])
            if (not np.isfinite(Tbeat)) or Tbeat <= 0:
                continue

            for k in range(n_branches):
                for r in range(n_radii):
                    v = v_block[:, b, k, r]
                    if v.size < 2 or not np.all(np.isfinite(v)):
                        continue

                    Vf = np.fft.rfft(v) / float(v.size)
                    if Vf.size < H + 1:
                        continue

                    V_hbkr[:, b, k, r] = Vf[: H + 1]
                    valid_waveform_bkr_mask[b, k, r] = True

        return {
            "V_hbkr": V_hbkr,
            "H": H,
            "valid_waveform_bkr_mask": valid_waveform_bkr_mask,
        }

    def _normalized_harmonics(self, V_hbkr: np.ndarray) -> dict:
        H = int(V_hbkr.shape[0] - 1)
        n_beats, n_branches, n_radii = V_hbkr.shape[1:]

        if H < 2:
            c_hbkr = self._complex_nan((0, n_beats, n_branches, n_radii))
            valid_c_hbkr_mask = np.zeros((0, n_beats, n_branches, n_radii), dtype=bool)
            harmonics = np.asarray([], dtype=int)
            rel_amp_hbkr = np.full((0, n_beats, n_branches, n_radii), np.nan, dtype=float)
            return {
                "c_hbkr": c_hbkr,
                "valid_c_hbkr_mask": valid_c_hbkr_mask,
                "harmonics": harmonics,
                "rel_amp_hbkr": rel_amp_hbkr,
            }

        V1_bkr = V_hbkr[1]
        valid_den_bkr = self._isfinite_complex(V1_bkr) & (np.abs(V1_bkr) > self.eps)

        c_hbkr = self._complex_nan((H - 1, n_beats, n_branches, n_radii))
        valid_c_hbkr_mask = np.zeros((H - 1, n_beats, n_branches, n_radii), dtype=bool)
        rel_amp_hbkr = np.full((H - 1, n_beats, n_branches, n_radii), np.nan, dtype=float)

        for hi, h in enumerate(range(2, H + 1)):
            Vh_bkr = V_hbkr[h]
            finite_h_bkr = self._isfinite_complex(Vh_bkr) & self._isfinite_complex(V1_bkr)

            rel = np.full((n_beats, n_branches, n_radii), np.nan, dtype=float)
            rel[finite_h_bkr] = (
                np.abs(Vh_bkr[finite_h_bkr]) / (np.abs(V1_bkr[finite_h_bkr]) + self.eps)
            )
            rel_amp_hbkr[hi] = rel

            ok = finite_h_bkr & valid_den_bkr & (
                rel > float(self.higher_harmonic_rel_threshold)
            )
            valid_c_hbkr_mask[hi] = ok

            c_h = self._complex_nan((n_beats, n_branches, n_radii))
            c_h[ok] = Vh_bkr[ok] / V1_bkr[ok]
            c_hbkr[hi] = c_h

        harmonics = np.arange(2, H + 1, dtype=int)
        return {
            "c_hbkr": c_hbkr,
            "valid_c_hbkr_mask": valid_c_hbkr_mask,
            "harmonics": harmonics,
            "rel_amp_hbkr": rel_amp_hbkr,
        }

    # ----------------------------
    # Matrix / SVD
    # ----------------------------
    def _build_matrix_X(self, c_hbkr: np.ndarray, valid_c_hbkr_mask: np.ndarray) -> dict:
        """
        Option 1:
        - keep rows with at least min_valid_harmonics_per_row valid harmonics
        - fill invalid harmonic entries with 0 in X
        """
        if c_hbkr.shape[0] == 0:
            n_beats, n_branches, n_radii = c_hbkr.shape[1:]
            return {
                "row_valid_bkr_mask": np.zeros((n_beats, n_branches, n_radii), dtype=bool),
                "valid_h_count_over_bkr": np.zeros((n_beats, n_branches, n_radii), dtype=np.int32),
                "row_index_bkr": np.full((n_beats, n_branches, n_radii), -1, dtype=np.int32),
                "X_j_over_h": self._complex_nan((0, 0)),
            }

        n_h, n_beats, n_branches, n_radii = c_hbkr.shape

        valid_h_count_over_bkr = self._count_true(valid_c_hbkr_mask, axis=0)
        min_valid = int(max(1, min(self.min_valid_harmonics_per_row, n_h)))
        row_valid_bkr_mask = valid_h_count_over_bkr >= min_valid

        row_index_bkr = np.full((n_beats, n_branches, n_radii), -1, dtype=np.int32)
        n_rows = int(np.sum(row_valid_bkr_mask))
        if n_rows > 0:
            row_index_bkr[row_valid_bkr_mask] = np.arange(n_rows, dtype=np.int32)

        X_j_over_h = np.zeros((n_rows, n_h), dtype=np.complex128)
        if n_rows > 0:
            for hi in range(n_h):
                vals = c_hbkr[hi][row_valid_bkr_mask]
                valid = valid_c_hbkr_mask[hi][row_valid_bkr_mask]
                col = np.zeros((n_rows,), dtype=np.complex128)
                col[valid] = vals[valid]
                X_j_over_h[:, hi] = col

        return {
            "row_valid_bkr_mask": row_valid_bkr_mask,
            "valid_h_count_over_bkr": valid_h_count_over_bkr,
            "row_index_bkr": row_index_bkr,
            "X_j_over_h": X_j_over_h,
        }

    def _rank1_svd(self, X_j_over_h: np.ndarray) -> dict:
        X_j_over_h = np.asarray(X_j_over_h, dtype=np.complex128)
        n_rows = int(X_j_over_h.shape[0])
        n_h = int(X_j_over_h.shape[1]) if X_j_over_h.ndim == 2 else 0

        if n_rows == 0 or n_h == 0:
            return {
                "singular_values": np.full((0,), np.nan, dtype=float),
                "sigma_1": np.nan,
                "p1_over_j": self._complex_nan((0,)),
                "q1_over_h": self._complex_nan((0,)),
                "Xhat1_j_over_h": self._complex_nan((n_rows, n_h)),
                "eta_1": np.nan,
            }

        U, s, Vh = np.linalg.svd(X_j_over_h, full_matrices=False)
        singular_values = np.asarray(s, dtype=float)

        sigma_1 = float(singular_values[0]) if singular_values.size > 0 else np.nan
        p1_over_j = (
            U[:, 0].astype(np.complex128)
            if U.shape[1] > 0
            else self._complex_nan((n_rows,))
        )
        q1_over_h = (
            np.conjugate(Vh[0, :]).astype(np.complex128)
            if Vh.shape[0] > 0
            else self._complex_nan((n_h,))
        )

        if np.isfinite(sigma_1):
            Xhat1_j_over_h = sigma_1 * np.outer(p1_over_j, np.conjugate(q1_over_h))
        else:
            Xhat1_j_over_h = self._complex_nan((n_rows, n_h))

        denom = float(np.sum(np.abs(X_j_over_h) ** 2))
        if (not np.isfinite(denom)) or denom <= self.eps:
            eta_1 = np.nan
        else:
            num = float(np.sum(np.abs(X_j_over_h - Xhat1_j_over_h) ** 2))
            eta_1 = float(1.0 - num / denom)

        return {
            "singular_values": singular_values,
            "sigma_1": sigma_1,
            "p1_over_j": p1_over_j,
            "q1_over_h": q1_over_h,
            "Xhat1_j_over_h": Xhat1_j_over_h,
            "eta_1": eta_1,
        }

    def _mode_scores(
        self, row_valid_bkr_mask: np.ndarray, sigma_1: float, p1_over_j: np.ndarray
    ) -> dict:
        n_beats, n_branches, n_radii = row_valid_bkr_mask.shape

        xi_bkr = self._complex_nan((n_beats, n_branches, n_radii))
        if p1_over_j.size > 0 and np.isfinite(sigma_1):
            xi_j = sigma_1 * p1_over_j
            xi_bkr[row_valid_bkr_mask] = xi_j

        xibar_kr_over_b = self._complex_mean(xi_bkr, axis=(1, 2))
        valid_kr_count_over_b = self._count_true(
            self._isfinite_complex(xi_bkr), axis=(1, 2)
        )

        xibar_b_over_kr = self._complex_mean(xi_bkr, axis=0)
        valid_b_count_over_kr = self._count_true(
            self._isfinite_complex(xi_bkr), axis=0
        )

        H_b_xi, valid_b_count_for_H_b_xi = self._safe_cv_abs(
            xibar_kr_over_b, self.eps
        )
        H_kr_xi, valid_kr_count_for_H_kr_xi = self._safe_cv_abs(
            xibar_b_over_kr.reshape(-1), self.eps
        )

        return {
            "xi_bkr": xi_bkr,
            "xibar_kr_over_b": xibar_kr_over_b,
            "xibar_b_over_kr": xibar_b_over_kr,
            "valid_kr_count_over_b": valid_kr_count_over_b,
            "valid_b_count_over_kr": valid_b_count_over_kr,
            "H_b_xi": H_b_xi,
            "H_kr_xi": H_kr_xi,
            "valid_b_count_for_H_b_xi": valid_b_count_for_H_b_xi,
            "valid_kr_count_for_H_kr_xi": valid_kr_count_for_H_kr_xi,
        }

    # ----------------------------
    # Tensor / rank-1 CP
    # ----------------------------
    def _weighted_projection_lambda(
        self,
        chi_hbkr: np.ndarray,
        W_hbkr: np.ndarray,
        z_over_h: np.ndarray,
        u_over_b: np.ndarray,
        v_over_k: np.ndarray,
        w_over_r: np.ndarray,
    ) -> complex:
        A_hbkr = (
            z_over_h[:, None, None, None]
            * u_over_b[None, :, None, None]
            * v_over_k[None, None, :, None]
            * w_over_r[None, None, None, :]
        )
        num = np.sum(np.where(W_hbkr, chi_hbkr * np.conjugate(A_hbkr), 0.0 + 0.0j))
        den = float(np.sum(np.where(W_hbkr, np.abs(A_hbkr) ** 2, 0.0)))
        if (not np.isfinite(den)) or den <= self.eps:
            return np.complex128(np.nan + 1j * np.nan)
        return np.complex128(num / den)

    def _weighted_eta(
        self, chi_hbkr: np.ndarray, W_hbkr: np.ndarray, chat1_hbkr: np.ndarray
    ) -> float:
        denom = float(np.sum(np.where(W_hbkr, np.abs(chi_hbkr) ** 2, 0.0)))
        if (not np.isfinite(denom)) or denom <= self.eps:
            return np.nan
        num = float(np.sum(np.where(W_hbkr, np.abs(chi_hbkr - chat1_hbkr) ** 2, 0.0)))
        return float(1.0 - num / denom)

    def _update_rank1_factor_h(
        self,
        chi_hbkr: np.ndarray,
        W_hbkr: np.ndarray,
        u_over_b: np.ndarray,
        v_over_k: np.ndarray,
        w_over_r: np.ndarray,
    ) -> np.ndarray:
        other = (
            u_over_b[None, :, None, None]
            * v_over_k[None, None, :, None]
            * w_over_r[None, None, None, :]
        )
        num = np.sum(
            np.where(W_hbkr, chi_hbkr * np.conjugate(other), 0.0 + 0.0j),
            axis=(1, 2, 3),
        )
        den = np.sum(np.where(W_hbkr, np.abs(other) ** 2, 0.0), axis=(1, 2, 3))
        z = np.zeros_like(num, dtype=np.complex128)
        ok = den > self.eps
        z[ok] = num[ok] / den[ok]
        return self._normalize_complex_vector(z, self.eps)

    def _update_rank1_factor_b(
        self,
        chi_hbkr: np.ndarray,
        W_hbkr: np.ndarray,
        z_over_h: np.ndarray,
        v_over_k: np.ndarray,
        w_over_r: np.ndarray,
    ) -> np.ndarray:
        other = (
            z_over_h[:, None, None, None]
            * v_over_k[None, None, :, None]
            * w_over_r[None, None, None, :]
        )
        num = np.sum(
            np.where(W_hbkr, chi_hbkr * np.conjugate(other), 0.0 + 0.0j),
            axis=(0, 2, 3),
        )
        den = np.sum(np.where(W_hbkr, np.abs(other) ** 2, 0.0), axis=(0, 2, 3))
        u = np.zeros_like(num, dtype=np.complex128)
        ok = den > self.eps
        u[ok] = num[ok] / den[ok]
        return self._normalize_complex_vector(u, self.eps)

    def _update_rank1_factor_k(
        self,
        chi_hbkr: np.ndarray,
        W_hbkr: np.ndarray,
        z_over_h: np.ndarray,
        u_over_b: np.ndarray,
        w_over_r: np.ndarray,
    ) -> np.ndarray:
        other = (
            z_over_h[:, None, None, None]
            * u_over_b[None, :, None, None]
            * w_over_r[None, None, None, :]
        )
        num = np.sum(
            np.where(W_hbkr, chi_hbkr * np.conjugate(other), 0.0 + 0.0j),
            axis=(0, 1, 3),
        )
        den = np.sum(np.where(W_hbkr, np.abs(other) ** 2, 0.0), axis=(0, 1, 3))
        v = np.zeros_like(num, dtype=np.complex128)
        ok = den > self.eps
        v[ok] = num[ok] / den[ok]
        return self._normalize_complex_vector(v, self.eps)

    def _update_rank1_factor_r(
        self,
        chi_hbkr: np.ndarray,
        W_hbkr: np.ndarray,
        z_over_h: np.ndarray,
        u_over_b: np.ndarray,
        v_over_k: np.ndarray,
    ) -> np.ndarray:
        other = (
            z_over_h[:, None, None, None]
            * u_over_b[None, :, None, None]
            * v_over_k[None, None, :, None]
        )
        num = np.sum(
            np.where(W_hbkr, chi_hbkr * np.conjugate(other), 0.0 + 0.0j),
            axis=(0, 1, 2),
        )
        den = np.sum(np.where(W_hbkr, np.abs(other) ** 2, 0.0), axis=(0, 1, 2))
        w = np.zeros_like(num, dtype=np.complex128)
        ok = den > self.eps
        w[ok] = num[ok] / den[ok]
        return self._normalize_complex_vector(w, self.eps)

    def _rank1_cp_tensor(self, c_hbkr: np.ndarray, valid_c_hbkr_mask: np.ndarray) -> dict:
        chi_hbkr = np.asarray(c_hbkr, dtype=np.complex128)
        W_hbkr = np.asarray(valid_c_hbkr_mask, dtype=bool)

        n_h, n_beats, n_branches, n_radii = chi_hbkr.shape

        if n_h == 0 or np.sum(W_hbkr) == 0:
            return {
                "valid_bkr_count_over_h": np.zeros((n_h,), dtype=np.int32),
                "valid_hkr_count_over_b": np.zeros((n_beats,), dtype=np.int32),
                "valid_hbr_count_over_k": np.zeros((n_branches,), dtype=np.int32),
                "valid_hbk_count_over_r": np.zeros((n_radii,), dtype=np.int32),
                "lambda_1_cp": np.complex128(np.nan + 1j * np.nan),
                "z_over_h": self._complex_nan((n_h,)),
                "u_over_b": self._complex_nan((n_beats,)),
                "v_over_k": self._complex_nan((n_branches,)),
                "w_over_r": self._complex_nan((n_radii,)),
                "chat1_hbkr": self._complex_nan((n_h, n_beats, n_branches, n_radii)),
                "eta_1_cp": np.nan,
                "cp_n_iter": 0,
                "cp_converged": 0,
                "cp_final_rel_change": np.nan,
            }

        chi_obs = np.where(W_hbkr, chi_hbkr, 0.0 + 0.0j)

        valid_bkr_count_over_h = self._count_true(W_hbkr, axis=(1, 2, 3))
        valid_hkr_count_over_b = self._count_true(W_hbkr, axis=(0, 2, 3))
        valid_hbr_count_over_k = self._count_true(W_hbkr, axis=(0, 1, 3))
        valid_hbk_count_over_r = self._count_true(W_hbkr, axis=(0, 1, 2))

        z_over_h = self._leading_left_singular_vector_or_uniform(
            chi_obs.reshape(n_h, -1)
        )
        u_over_b = self._leading_left_singular_vector_or_uniform(
            np.transpose(chi_obs, (1, 0, 2, 3)).reshape(n_beats, -1)
        )
        v_over_k = self._leading_left_singular_vector_or_uniform(
            np.transpose(chi_obs, (2, 0, 1, 3)).reshape(n_branches, -1)
        )
        w_over_r = self._leading_left_singular_vector_or_uniform(
            np.transpose(chi_obs, (3, 0, 1, 2)).reshape(n_radii, -1)
        )

        prev_chat1_hbkr = None
        cp_converged = 0
        cp_final_rel_change = np.nan
        cp_n_iter = 0

        for it in range(1, self.cp_max_iter + 1):
            cp_n_iter = it

            z_over_h = self._update_rank1_factor_h(
                chi_hbkr, W_hbkr, u_over_b, v_over_k, w_over_r
            )
            u_over_b = self._update_rank1_factor_b(
                chi_hbkr, W_hbkr, z_over_h, v_over_k, w_over_r
            )
            v_over_k = self._update_rank1_factor_k(
                chi_hbkr, W_hbkr, z_over_h, u_over_b, w_over_r
            )
            w_over_r = self._update_rank1_factor_r(
                chi_hbkr, W_hbkr, z_over_h, u_over_b, v_over_k
            )

            lambda_1_cp = self._weighted_projection_lambda(
                chi_hbkr, W_hbkr, z_over_h, u_over_b, v_over_k, w_over_r
            )

            chat1_hbkr = (
                lambda_1_cp
                * z_over_h[:, None, None, None]
                * u_over_b[None, :, None, None]
                * v_over_k[None, None, :, None]
                * w_over_r[None, None, None, :]
            )

            denom = float(
                np.sqrt(np.sum(np.where(W_hbkr, np.abs(chi_hbkr) ** 2, 0.0)))
            )
            if prev_chat1_hbkr is None or (not np.isfinite(denom)) or denom <= self.eps:
                rel_change = np.nan
            else:
                diff = float(
                    np.sqrt(
                        np.sum(
                            np.where(
                                W_hbkr, np.abs(chat1_hbkr - prev_chat1_hbkr) ** 2, 0.0
                            )
                        )
                    )
                )
                rel_change = diff / (denom + self.eps)

            prev_chat1_hbkr = chat1_hbkr.copy()
            cp_final_rel_change = rel_change

            if np.isfinite(rel_change) and rel_change < self.cp_tol:
                cp_converged = 1
                break

        lambda_1_cp = self._weighted_projection_lambda(
            chi_hbkr, W_hbkr, z_over_h, u_over_b, v_over_k, w_over_r
        )
        chat1_hbkr = (
            lambda_1_cp
            * z_over_h[:, None, None, None]
            * u_over_b[None, :, None, None]
            * v_over_k[None, None, :, None]
            * w_over_r[None, None, None, :]
        )
        eta_1_cp = self._weighted_eta(chi_hbkr, W_hbkr, chat1_hbkr)

        return {
            "valid_bkr_count_over_h": valid_bkr_count_over_h,
            "valid_hkr_count_over_b": valid_hkr_count_over_b,
            "valid_hbr_count_over_k": valid_hbr_count_over_k,
            "valid_hbk_count_over_r": valid_hbk_count_over_r,
            "lambda_1_cp": lambda_1_cp,
            "z_over_h": z_over_h,
            "u_over_b": u_over_b,
            "v_over_k": v_over_k,
            "w_over_r": w_over_r,
            "chat1_hbkr": chat1_hbkr,
            "eta_1_cp": eta_1_cp,
            "cp_n_iter": cp_n_iter,
            "cp_converged": cp_converged,
            "cp_final_rel_change": cp_final_rel_change,
        }

    # ----------------------------
    # Main block computation
    # ----------------------------
    def _compute_representation_metrics(
        self, v_block: np.ndarray, T_vec: np.ndarray
    ) -> dict:
        harm = self._harmonic_coefficients_block(v_block, T_vec)
        V_hbkr = harm["V_hbkr"]
        H = harm["H"]
        valid_waveform_bkr_mask = harm["valid_waveform_bkr_mask"]

        norm = self._normalized_harmonics(V_hbkr)
        c_hbkr = norm["c_hbkr"]
        valid_c_hbkr_mask = norm["valid_c_hbkr_mask"]
        harmonics = norm["harmonics"]
        rel_amp_hbkr = norm["rel_amp_hbkr"]

        mat = self._build_matrix_X(c_hbkr, valid_c_hbkr_mask)
        row_valid_bkr_mask = mat["row_valid_bkr_mask"]
        valid_h_count_over_bkr = mat["valid_h_count_over_bkr"]
        row_index_bkr = mat["row_index_bkr"]
        X_j_over_h = mat["X_j_over_h"]

        svd = self._rank1_svd(X_j_over_h)
        mode = self._mode_scores(
            row_valid_bkr_mask,
            svd["sigma_1"],
            svd["p1_over_j"],
        )

        cp = self._rank1_cp_tensor(c_hbkr, valid_c_hbkr_mask)

        return {
            "H": int(H),
            "harmonics": harmonics,
            "valid_waveform_bkr_mask": valid_waveform_bkr_mask.astype(np.uint8),
            "valid_c_hbkr_mask": valid_c_hbkr_mask.astype(np.uint8),
            "V_hbkr": V_hbkr,
            "c_hbkr": c_hbkr,
            "rel_amp_hbkr": rel_amp_hbkr,
            "valid_h_count_over_bkr": valid_h_count_over_bkr,
            "row_valid_bkr_mask": row_valid_bkr_mask.astype(np.uint8),
            "row_index_bkr": row_index_bkr,
            "X_j_over_h": X_j_over_h,
            "singular_values": svd["singular_values"],
            "sigma_1": svd["sigma_1"],
            "p1_over_j": svd["p1_over_j"],
            "q1_over_h": svd["q1_over_h"],
            "Xhat1_j_over_h": svd["Xhat1_j_over_h"],
            "eta_1": svd["eta_1"],
            "xi_bkr": mode["xi_bkr"],
            "xibar_kr_over_b": mode["xibar_kr_over_b"],
            "xibar_b_over_kr": mode["xibar_b_over_kr"],
            "valid_kr_count_over_b": mode["valid_kr_count_over_b"],
            "valid_b_count_over_kr": mode["valid_b_count_over_kr"],
            "H_b_xi": mode["H_b_xi"],
            "H_kr_xi": mode["H_kr_xi"],
            "valid_b_count_for_H_b_xi": mode["valid_b_count_for_H_b_xi"],
            "valid_kr_count_for_H_kr_xi": mode["valid_kr_count_for_H_kr_xi"],
            "valid_bkr_count_over_h": cp["valid_bkr_count_over_h"],
            "valid_hkr_count_over_b": cp["valid_hkr_count_over_b"],
            "valid_hbr_count_over_k": cp["valid_hbr_count_over_k"],
            "valid_hbk_count_over_r": cp["valid_hbk_count_over_r"],
            "lambda_1_cp": cp["lambda_1_cp"],
            "z_over_h": cp["z_over_h"],
            "u_over_b": cp["u_over_b"],
            "v_over_k": cp["v_over_k"],
            "w_over_r": cp["w_over_r"],
            "chat1_hbkr": cp["chat1_hbkr"],
            "eta_1_cp": cp["eta_1_cp"],
            "cp_n_iter": cp["cp_n_iter"],
            "cp_converged": cp["cp_converged"],
            "cp_final_rel_change": cp["cp_final_rel_change"],
        }

    # ----------------------------
    # Writing helpers
    # ----------------------------
    def _pack_split_complex(
        self,
        metrics: dict,
        path: str,
        z: np.ndarray,
        attrs_common: dict,
    ) -> None:
        metrics[f"{path}_real"] = with_attrs(
            np.asarray(np.real(z), dtype=np.float32), attrs_common
        )
        metrics[f"{path}_imag"] = with_attrs(
            np.asarray(np.imag(z), dtype=np.float32), attrs_common
        )

    def _pack_representation_outputs(
        self, metrics: dict, vessel_prefix: str, out: dict
    ) -> None:
        base = f"{vessel_prefix}/by_segment/raw"

        metrics[f"{base}/params/H_MAX"] = np.asarray(self.H_MAX, dtype=np.int32)
        metrics[f"{base}/params/eps"] = np.asarray(self.eps, dtype=np.float32)
        metrics[f"{base}/params/higher_harmonic_rel_threshold"] = np.asarray(
            self.higher_harmonic_rel_threshold, dtype=np.float32
        )
        metrics[f"{base}/params/min_valid_harmonics_per_row"] = np.asarray(
            self.min_valid_harmonics_per_row, dtype=np.int32
        )
        metrics[f"{base}/params/cp_max_iter"] = np.asarray(
            self.cp_max_iter, dtype=np.int32
        )
        metrics[f"{base}/params/cp_tol"] = np.asarray(self.cp_tol, dtype=np.float32)

        metrics[f"{base}/axes/harmonics"] = with_attrs(
            np.asarray(out["harmonics"], dtype=np.int32),
            {
                "definition": [
                    "Higher-harmonic index array corresponding to h=2..H for normalized coefficients."
                ]
            },
        )
        metrics[f"{base}/axes/j_valid"] = with_attrs(
            np.arange(out["X_j_over_h"].shape[0], dtype=np.int32),
            {"definition": ["Valid row index j for X_j_over_h and rank-1 SVD quantities."]},
        )

        metrics[f"{base}/masks/valid_waveform_bkr_mask"] = with_attrs(
            np.asarray(out["valid_waveform_bkr_mask"], dtype=np.uint8),
            {
                "definition": [
                    "1 where the input beat/segment waveform was finite and FFT was computed."
                ]
            },
        )
        metrics[f"{base}/masks/valid_c_hbkr_mask"] = with_attrs(
            np.asarray(out["valid_c_hbkr_mask"], dtype=np.uint8),
            {
                "definition": [
                    "1 where c_hbkr = V_hbkr / V_1bkr satisfied the higher-harmonic relative validity criterion."
                ]
            },
        )
        metrics[f"{base}/masks/row_valid_bkr_mask"] = with_attrs(
            np.asarray(out["row_valid_bkr_mask"], dtype=np.uint8),
            {
                "definition": [
                    "1 where the beat-location pair (b,k,r) has at least min_valid_harmonics_per_row valid higher harmonics and contributes one row to X."
                ]
            },
        )

        self._pack_split_complex(
            metrics,
            f"{base}/harmonics/V_hbkr",
            out["V_hbkr"],
            {
                "definition": ["Complex Fourier coefficients V_hbkr for harmonics h=0..H."],
                "layout": ["(harmonic, beat, branch, radius)"],
            },
        )

        self._pack_split_complex(
            metrics,
            f"{base}/normalized/c_hbkr",
            out["c_hbkr"],
            {
                "definition": [r"Normalized higher harmonics c_hbkr = V_hbkr / V_1bkr for h=2..H."],
                "layout": ["(higher_harmonic, beat, branch, radius)"],
            },
        )
        metrics[f"{base}/normalized/rel_amp_hbkr"] = with_attrs(
            np.asarray(out["rel_amp_hbkr"], dtype=np.float32),
            {
                "definition": [r"Relative higher-harmonic amplitude |V_hbkr| / |V_1bkr| used in the validity rule."],
                "layout": ["(higher_harmonic, beat, branch, radius)"],
            },
        )
        metrics[f"{base}/normalized/valid_h_count_over_bkr"] = with_attrs(
            np.asarray(out["valid_h_count_over_bkr"], dtype=np.int32),
            {
                "definition": [
                    "Number of valid higher harmonics at each beat-location pair (b,k,r)."
                ],
                "layout": ["(beat, branch, radius)"],
            },
        )

        metrics[f"{base}/matrix/row_index_bkr"] = with_attrs(
            np.asarray(out["row_index_bkr"], dtype=np.int32),
            {
                "definition": [
                    "Row index j corresponding to each valid (b,k,r) in X_j_over_h; invalid entries are -1."
                ],
                "layout": ["(beat, branch, radius)"],
            },
        )
        self._pack_split_complex(
            metrics,
            f"{base}/matrix/X_j_over_h",
            out["X_j_over_h"],
            {
                "definition": [
                    r"Matrix X with rows j corresponding to beat-location pairs having at least min_valid_harmonics_per_row valid harmonics; invalid harmonic entries are zero-filled."
                ],
                "layout": ["(j_valid, higher_harmonic)"],
            },
        )

        metrics[f"{base}/svd/singular_values"] = with_attrs(
            np.asarray(out["singular_values"], dtype=np.float32),
            {"definition": ["Singular values of X in descending order."]},
        )
        metrics[f"{base}/svd/sigma_1"] = with_attrs(
            np.asarray(out["sigma_1"], dtype=np.float32),
            {"definition": [r"Leading singular value \sigma_1 of X."]},
        )
        self._pack_split_complex(
            metrics,
            f"{base}/svd/p1_over_j",
            out["p1_over_j"],
            {
                "definition": [r"Leading left singular vector p_1 indexed by valid row j."],
                "layout": ["(j_valid,)"],
            },
        )
        self._pack_split_complex(
            metrics,
            f"{base}/svd/q1_over_h",
            out["q1_over_h"],
            {
                "definition": [r"Leading right singular vector q_1 indexed by higher harmonic h=2..H."],
                "layout": ["(higher_harmonic,)"],
            },
        )
        self._pack_split_complex(
            metrics,
            f"{base}/svd/Xhat1_j_over_h",
            out["Xhat1_j_over_h"],
            {
                "definition": [r"Rank-1 approximation \hat X_1 = \sigma_1 p_1 q_1^*."],
                "layout": ["(j_valid, higher_harmonic)"],
            },
        )
        metrics[f"{base}/svd/eta_1"] = with_attrs(
            np.asarray(out["eta_1"], dtype=np.float32),
            {
                "definition": [
                    r"Explained fraction \eta_1 = 1 - ||X-\hat X_1||_F^2 / ||X||_F^2."
                ]
            },
        )

        self._pack_split_complex(
            metrics,
            f"{base}/mode_scores/xi_bkr",
            out["xi_bkr"],
            {
                "definition": [r"Leading mode score \xi_{bkr} = \sigma_1 P_{j1}, mapped back to (b,k,r)."],
                "layout": ["(beat, branch, radius)"],
            },
        )
        self._pack_split_complex(
            metrics,
            f"{base}/mode_scores/xibar_kr_over_b",
            out["xibar_kr_over_b"],
            {
                "definition": [r"Beat-varying mode score \bar\xi^{(kr)}_b = mean_{k,r}(\xi_{bkr})."],
                "layout": ["(beat,)"],
            },
        )
        self._pack_split_complex(
            metrics,
            f"{base}/mode_scores/xibar_b_over_kr",
            out["xibar_b_over_kr"],
            {
                "definition": [r"Location-varying mode score \bar\xi^{(b)}_{kr} = mean_b(\xi_{bkr})."],
                "layout": ["(branch, radius)"],
            },
        )
        metrics[f"{base}/mode_scores/valid_kr_count_over_b"] = with_attrs(
            np.asarray(out["valid_kr_count_over_b"], dtype=np.int32),
            {
                "definition": [
                    "Number of valid locations contributing to xibar_kr_over_b at each beat b."
                ],
                "layout": ["(beat,)"],
            },
        )
        metrics[f"{base}/mode_scores/valid_b_count_over_kr"] = with_attrs(
            np.asarray(out["valid_b_count_over_kr"], dtype=np.int32),
            {
                "definition": [
                    "Number of valid beats contributing to xibar_b_over_kr at each location (k,r)."
                ],
                "layout": ["(branch, radius)"],
            },
        )
        metrics[f"{base}/mode_scores/H_b_xi"] = with_attrs(
            np.asarray(out["H_b_xi"], dtype=np.float32),
            {
                "definition": [
                    r"Heterogeneity index H^{(b)}_\xi = CV_b(|\bar\xi^{(kr)}_b|)."
                ]
            },
        )
        metrics[f"{base}/mode_scores/H_kr_xi"] = with_attrs(
            np.asarray(out["H_kr_xi"], dtype=np.float32),
            {
                "definition": [
                    r"Heterogeneity index H^{(kr)}_\xi = CV_{k,r}(|\bar\xi^{(b)}_{kr}|)."
                ]
            },
        )
        metrics[f"{base}/mode_scores/valid_b_count_for_H_b_xi"] = with_attrs(
            np.asarray(out["valid_b_count_for_H_b_xi"], dtype=np.int32),
            {
                "definition": [
                    "Number of valid beat entries used to compute H_b_xi."
                ]
            },
        )
        metrics[f"{base}/mode_scores/valid_kr_count_for_H_kr_xi"] = with_attrs(
            np.asarray(out["valid_kr_count_for_H_kr_xi"], dtype=np.int32),
            {
                "definition": [
                    "Number of valid location entries used to compute H_kr_xi."
                ]
            },
        )

        metrics[f"{base}/tensor/valid_bkr_count_over_h"] = with_attrs(
            np.asarray(out["valid_bkr_count_over_h"], dtype=np.int32),
            {
                "definition": [
                    "Number of valid (b,k,r) entries contributing to each harmonic-mode factor z_h."
                ],
                "layout": ["(higher_harmonic,)"],
            },
        )
        metrics[f"{base}/tensor/valid_hkr_count_over_b"] = with_attrs(
            np.asarray(out["valid_hkr_count_over_b"], dtype=np.int32),
            {
                "definition": [
                    "Number of valid (h,k,r) entries contributing to each beat-mode factor u_b."
                ],
                "layout": ["(beat,)"],
            },
        )
        metrics[f"{base}/tensor/valid_hbr_count_over_k"] = with_attrs(
            np.asarray(out["valid_hbr_count_over_k"], dtype=np.int32),
            {
                "definition": [
                    "Number of valid (h,b,r) entries contributing to each branch-mode factor v_k."
                ],
                "layout": ["(branch,)"],
            },
        )
        metrics[f"{base}/tensor/valid_hbk_count_over_r"] = with_attrs(
            np.asarray(out["valid_hbk_count_over_r"], dtype=np.int32),
            {
                "definition": [
                    "Number of valid (h,b,k) entries contributing to each radius-mode factor w_r."
                ],
                "layout": ["(radius,)"],
            },
        )

        self._pack_split_complex(
            metrics,
            f"{base}/tensor/lambda_1_cp",
            np.asarray(out["lambda_1_cp"]),
            {
                "definition": [r"Rank-1 CP scalar weight \lambda for \hat\chi_1 = \lambda z \otimes u \otimes v \otimes w."],
            },
        )
        self._pack_split_complex(
            metrics,
            f"{base}/tensor/z_over_h",
            out["z_over_h"],
            {
                "definition": [r"Harmonic-mode factor z_h in the rank-1 CP model."],
                "layout": ["(higher_harmonic,)"],
            },
        )
        self._pack_split_complex(
            metrics,
            f"{base}/tensor/u_over_b",
            out["u_over_b"],
            {
                "definition": [r"Beat-mode factor u_b in the rank-1 CP model."],
                "layout": ["(beat,)"],
            },
        )
        self._pack_split_complex(
            metrics,
            f"{base}/tensor/v_over_k",
            out["v_over_k"],
            {
                "definition": [r"Branch-mode factor v_k in the rank-1 CP model."],
                "layout": ["(branch,)"],
            },
        )
        self._pack_split_complex(
            metrics,
            f"{base}/tensor/w_over_r",
            out["w_over_r"],
            {
                "definition": [r"Radius-mode factor w_r in the rank-1 CP model."],
                "layout": ["(radius,)"],
            },
        )
        self._pack_split_complex(
            metrics,
            f"{base}/tensor/chat1_hbkr",
            out["chat1_hbkr"],
            {
                "definition": [r"Rank-1 tensor approximation \hat\chi_1 = \lambda z \otimes u \otimes v \otimes w."],
                "layout": ["(higher_harmonic, beat, branch, radius)"],
            },
        )
        metrics[f"{base}/tensor/eta_1_cp"] = with_attrs(
            np.asarray(out["eta_1_cp"], dtype=np.float32),
            {
                "definition": [
                    r"Explained fraction \eta^{CP}_1 = 1 - ||\chi-\hat\chi_1||_F^2 / ||\chi||_F^2 evaluated on observed entries."
                ]
            },
        )
        metrics[f"{base}/tensor/cp_n_iter"] = with_attrs(
            np.asarray(out["cp_n_iter"], dtype=np.int32),
            {"definition": ["Number of ALS iterations performed for the rank-1 CP fit."]},
        )
        metrics[f"{base}/tensor/cp_converged"] = with_attrs(
            np.asarray(out["cp_converged"], dtype=np.uint8),
            {"definition": ["1 if the rank-1 CP ALS loop met the convergence tolerance."]},
        )
        metrics[f"{base}/tensor/cp_final_rel_change"] = with_attrs(
            np.asarray(out["cp_final_rel_change"], dtype=np.float32),
            {"definition": ["Final relative reconstruction change on observed entries in the CP ALS loop."]},
        )

    def _pack_vessel_outputs(
        self,
        metrics: dict,
        vessel_prefix: str,
        v_seg: np.ndarray,
        T: np.ndarray,
    ) -> None:
        if v_seg.ndim != 4:
            raise ValueError(
                f"Expected (n_t,n_beats,n_branches,n_radii), got {v_seg.shape}"
            )

        n_t, n_beats, n_branches, n_radii = v_seg.shape
        T_vec = self._ensure_beat_periods(T, n_beats)

        out = self._compute_representation_metrics(v_seg, T_vec)
        self._pack_representation_outputs(metrics, vessel_prefix, out)

        metrics[f"{vessel_prefix}/by_segment/params/n_t"] = np.asarray(
            n_t, dtype=np.int32
        )
        metrics[f"{vessel_prefix}/by_segment/params/n_beats"] = np.asarray(
            n_beats, dtype=np.int32
        )
        metrics[f"{vessel_prefix}/by_segment/params/n_branches"] = np.asarray(
            n_branches, dtype=np.int32
        )
        metrics[f"{vessel_prefix}/by_segment/params/n_radii"] = np.asarray(
            n_radii, dtype=np.int32
        )
        metrics[f"{vessel_prefix}/by_segment/params/n_valid_rows"] = np.asarray(
            out["X_j_over_h"].shape[0], dtype=np.int32
        )
        metrics[f"{vessel_prefix}/by_segment/params/n_higher_harmonics"] = np.asarray(
            out["X_j_over_h"].shape[1] if out["X_j_over_h"].ndim == 2 else 0,
            dtype=np.int32,
        )

    def run(self, h5file) -> ProcessResult:
        T = np.asarray(h5file[self.T_input])
        metrics = {}

        vessel_configs = [
            {
                "prefix": "artery",
                "v_raw_segment_input": self.v_raw_segment_input_artery,
            },
            {
                "prefix": "vein",
                "v_raw_segment_input": self.v_raw_segment_input_vein,
            },
        ]

        for cfg in vessel_configs:
            vessel_prefix = cfg["prefix"]

            if cfg["v_raw_segment_input"] in h5file:
                v_raw_seg = np.asarray(h5file[cfg["v_raw_segment_input"]])
                self._pack_vessel_outputs(metrics, vessel_prefix, v_raw_seg, T)

        return ProcessResult(metrics=metrics)
