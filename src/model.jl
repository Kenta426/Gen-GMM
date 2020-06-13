module Models
    # define generative functions
    include("../src/dirichlet.jl")
    using Gen

    # map gaussian distribution
    @gen (static) function generate_normal(mean::Float64, scale::Float64)
        return @trace(normal(mean, scale), :normal)
    end
    map_normal = Map(generate_normal)
    load_generated_functions()
    
    # multinomiral with dirichlet prior
    @gen (static) function DirichletMulti(alpha, K::Int64, N::Int64)
        base = fill(alpha/K, K)
        pi = dirichlet(base)
        obj = [@trace(categorical(pi), i=>:c) for i in 1:N]
        return obj
    end

    # gaussian mixture model given list of Int for the component class
    @gen (static) function GMM(c_label)
        n = length(c_label)
        k = length(Set(c_label))
        # mean
        muVar = @trace(inv_gamma(4.0, 4.0), :muVar)
        mu = @trace(map_normal(fill(0, k), fill(muVar, k)), :mu)
        # generate X
        xNoise = @trace(inv_gamma(4.0, 4.0), :xNoise)
        X = @trace(map_normal(mu[c_label], fill(xNoise, n)), :X)
    end

    # finite gaussian mixture model with K components
    @gen (static) function FGMM(K::Int64, N::Int64)
        alpha = @trace(Gen.gamma(1, 1), :alpha) # prior for the concentration parameter
        class_label = @trace(DirichletMulti(alpha, K, N), :Cluster) # generate cluster labels
        X = @trace(GMM(class_label), :Outcome) # generate X given class labels
        return X
    end
end
