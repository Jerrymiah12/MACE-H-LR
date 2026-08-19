using Distributed
@everywhere using LinearAlgebra


# Function to construct reciprocal space Hamiltonian and overlap matrices
@everywhere function reciprocal_space_matrices(H_real, S_real, R, k)
    N = size(H_real)[3]  
    dim = size(H_real)[1] 
    H_k = zeros(Complex{Float64}, dim, dim)
    S_k = zeros(Complex{Float64}, dim, dim)

    # Building the reciprocal space matrices
    for i in 1:N
        phase_factor = exp(im * 2π * dot(k, R[:, i]))
        H_k .+= H_real[:, :, i] * phase_factor
        S_k .+= S_real[:, :, i] * phase_factor
    end

    return H_k, S_k
end


@everywhere function diagnalize_HS(H_real, S_real, R, k_points)
    bands = []
    tasks = []

    for k in k_points
        push!(tasks, @spawn begin
        H_k, S_k = reciprocal_space_matrices(H_real, S_real, R, k)
        H_k = Hermitian(H_k)
        S_k = Hermitian(S_k)
        eigvals(H_k, S_k)
        end)
    end

    tasks = fetch.(tasks)

    for task in tasks
        eigvals = task
        push!(bands, eigvals)
    end 

    return k_points, bands
end