## Computed Tomography
\label{sec:basics}

In X-ray CT, an object is illuminated by X-rays from multiple angles, with the intensity of radiation transmitted through the object measured by detectors the opposite side of the object. Through this method, we aim to reconstruct a spatial distribution of the linear attenuation coefficient, written $f(x,y)$ in the two-dimensional case, and with units of inverse length. This gives the probability per unit path length that an X-ray photon is removed from the beam by absorption or scattering. Therefore, assuming X-rays are monoenergetic, the transmitted intensity along a ray $L$ should obey the Beer-Lambert law,

$$
I=I_0 \exp\left(-\int_L f(x,y) \, ds\right),
$$

where $I_0$ is the intensity of radiation incident on the object, $I$ is the detected intensity, and $ds$ denotes integration along the ray. Taking logarithms therefore then gives the linearised projection

$$
P=-\log\left(\frac{I}{I_0}\right)=\int_Lf(x,y)\, ds,
$$

meaning log-normalised CT measurements are approximated by a line integral through the attenuation image. This approach assumes that beams are narrow, monoenergetic and non-diffracting, and neglects scattering, beam hardening and finite aperture effects, all of which contribute in clinical CT. However, it provides a tractable mathematical basis within which to consider reconstruction algorithms.

### Detector Geometries

There are three common CT detector geometries; parallel-beam, fan-beam, and cone-beam. Parallel-beam assumes we have a linear array of X-ray sources, each focussed solely on a detector placed on the other side of the sample. These rays are therefore parallel, as illustrated in Figure \ref{fig:geometries}a. This is impractical to implement clinically, but allows for tractable mathematics for the Radon transform and the Fourier slice theorem.

Fan-beam CT is often used in practice, and utilises a point source from which a fan-blade of emissions are measured by an line of detectors. Writing the source position as $S(\beta)$ for a source angle $\beta$, and $d(\beta, \gamma)$ as the unit direction of a ray labelled by fan angle $\gamma$ allows us to write a fan-beam projection as

$$
R_\beta(\gamma) = \int_0^\infty f(S(\beta)+sd(\beta, \gamma))\, ds,
$$

where we assume the sample has compact support such that $f(\mathbf{r})=0$ outside of the object.

Cone-beam CT extends this to three dimensions, again using a point source, but now a two-dimensional array of detectors with coordinates labelled $u$ and $v$, such that the cone-beam projection has the form

$$
R_\beta(u,v)=\int_0^\infty f(S(\beta)+sd(\beta, u,v))\, ds.
$$

Note, however, that by restricting the cone-beam geometry to a single strip of detectors recovers the fan-beam geometry.

### Line Integrals, Projections and Sinograms

For a fixed projection angle $\theta$, we can define a line in the image plane by

$$
x\cos \theta + y\sin \theta=t,
$$

\label{eq:line_t}

where $t$ is the perpendicular distance of the line from the origin, as shown in Figure \ref{fig:line_integral}. The integral over this line can then be denoted as

$$
P_\theta(t) = \int_{L_{(\theta,t)}} f(x,y) \, ds,
$$

or equivalently in Dirac delta form,

$$
P_\theta(t)=\int_{-\infty}^\infty \int_{-\infty}^\infty  f(x,y)\delta(x\cos \theta+ y\sin \theta-t) \, dx\, dy.
$$

Notably, the projection $P_\theta(t)$ is therefore also the Radon transform \cite{radon} of $f(x,y)$ at angle $\theta$. We collect a line or grid of these projections simultaneously, each corresponding to the same $\theta$ alignment, but different values of $t$. In a hypothetical parallel-beam scan, all of these $\theta$-aligned rays are parallel, whereas in a realistic fan-beam or cone-beam scan, the angle refers to the central projection angle, from which other rays diverge.

In practice, we sample these projection lines from $N_\theta$ different angular views, $\theta_1, \theta_2, \dots, \theta_{N_\theta}$, and collect the projections within $N_t$ detector bins corresponding to $t_1, t_2, \dots t_{N_t}$, forming an array of measurements called a sinogram, denoted

$$
Y_{j,i}=P_{\theta_i}(t_j), \quad i=1,\dots, N_\theta, \quad j=1,\dots, N_t.
$$

Here, each column is a line of projections at directed angle $\theta_i$, and each row corresponds to one detector bin. By vectorising the image of $f(x,y)$ as $\mathbf{x}\in \mathbb{R}^n$, and the sinogram as $\mathbf{y}\in \mathbb{R}^m$, the discretised CT forward model can be written as

$$
\mathbf{y}=A\mathbf{x}+\boldsymbol{\eta},
$$

\label{eq:forward-model}

where $A\in \mathbb{R}^{m \times n}$ is the discretised forward projection operator, $\boldsymbol{\eta}$ encodes noise and modelling errors, and $m=N_\theta N_t$ in 2D. We can then express this operator $A$ explicitly by denoting an image pixel $\ell$ by $\Omega_\ell$, and using the pixel basis function

$$  
\varphi_\ell(x,y)=  
\begin{cases}  
1, & (x,y)\in \Omega_\ell,\\ 
0, & \text{otherwise},  
\end{cases}  
$$

such that we can approximate the attenuation field as 

$$
f(x,y)\approx \sum_{\ell=1}^n x_\ell \varphi_\ell(x,y),
$$

where $x_\ell$ is the value of pixel $\ell$. Therefore, substituting into \eqref{eq:line_t} and then comparing with \eqref{eq:forward-model} gives

$$
\begin{align}
P_{\theta_i}(t_j)  
&\approx  
\sum_{\ell=1}^n x_\ell  
\int_{L_{(\theta_i,t_j)}} \varphi_\ell(x,y)\,ds\\
\implies A_{(j,i),\ell}&=\int_{L_{(\theta_i, t_j)}}\varphi_\ell(x,y)\, ds,
\end{align}
$$

so each row of $A$ corresponds to a single measured ray, with entries encoding the path length of that ray within each image pixel. This matrix is generally extremely large, and impractical to store or manipulate.

### The Fourier Slice Theorem

Using the perpendicular distance of a ray from the origin, \eqref{eq:line_t} and the distance along that ray given by

$$
s=-x\sin \theta+ y \cos\theta
$$

give the projection at angle $\theta$ as

$$
P_\theta(t)=
\int_{-\infty}^{\infty}
f(t\cos\theta-s\sin\theta,\;t\sin\theta+s\cos\theta)\,ds.
$$

Taking the Fourier transform of this with respect to $t$ then gives a Fourier slice,

$$
S_\theta(\omega)
=
\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}
f(t\cos\theta-s\sin\theta,\;t\sin\theta+s\cos\theta)
e^{-2\pi i \omega t}
\,ds\,dt,
$$

so converting to the $(x,y)$ coordinate system gives

$$
S_\theta(\omega)
=
\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}
f(x,y)e^{-2\pi i\omega(x\cos\theta+y\sin\theta)}
\,dx\,dy = F(\omega \cos \theta, \omega \sin \theta).
$$

Therefore, the Fourier transform of a projection gives the two-dimensional Fourier transform of the object along the radial line at the same angle, as illustrated in Figure \ref{fig:fourier_slice_theorem}.

### Dosage and Noise

The radiation dose imparted on a patient is directly proportional to the total energy deposited in the tissue. For this reason, it is also directly proportional to the energy of the photons used, the number of photons released at each view, and the number of views acquired. Dosage can therefore be reduced by decreasing any of these properties. However, these each decrease imaging quality.

Measurement noise is intrinsically linked to the intensity of radiation used. If $C_0$ photons are incident on the sample along a ray with projection value $P$, then we expect to detect

$$
\bar C = C_0 \exp(-P).
$$

As this is a counting experiment, the noise due to counting errors can be approximated as Poisson such that

$$
C\sim \operatorname{Poisson}(\bar C).
$$

The projection estimate,

$$
\hat P =-\log \left(\frac{C}{C_0}\right),
$$

so at high photon counts where $\operatorname{Poisson}(\bar C)\approx \mathcal{N}(\bar C, \bar C)$, Taylor expansion gives

$$
\hat P \approx P+\eta, \quad \eta \sim \mathcal N\left(0, \frac{1}{\bar C}\right),
$$

at low photon counts (intensities), however, we may have $C=0$, in which case $\hat P=-\log 0$, which is undefined. Furthermore, relative fluctuations are large, preventing Taylor expansion, and in general, the noise after the logarithm becomes non-gaussian, signal-dependent, biased and possibly undefined \cite{}, meaning this domain is often better to model with the Poisson likelihood. Decreasing intensity therefore increases the difficulty in modelling, and is not discussed in detail in this work. However, interesting advances in using diffusion models trained with these sorts of priors can be found at \cite{}.

Decreasing the number of views also decreases the dosage, but in accordance with the Fourier slice theorem, this also limits the angular coverage of the Fourier space corresponding to $f(x,y)$, with only a finite set of radial Fourier lines being directly sampled. The unsampled angular regions must therefore be inferred by a reconstruction algorithm or a prior, motivating the use of the learned image priors discussed in Section~\ref{sec:diffusion_models}. If using a direct reconstruction approach, however, this lack of information can lead to streak artefacts and poorly defined edges where high-frequency Fourier information is not available.