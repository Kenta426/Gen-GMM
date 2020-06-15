module Inference
    # gibbs samplers
    using Gen
    using Statistics

    # gibbs sampler for concenrtation parameter
    function alpha_gibbs(trace, k)
        alpha = trace[:alpha]
        obj = trace[:Cluster]
        a = 1 # prior for alpha
        b = 1 # prior for alpha
        n = length(obj)

        # augmentation
        eta = Gen.beta(alpha+1, n)
        pi = n/alpha
        pi = pi/(1+pi)
        s = float(bernoulli(pi))
        alpha = Gen.gamma(a+k-s, 1/(b-log(eta)))

        # update addr for alpha
        (trace, w, _, discard) = update(trace, get_args(trace), (), choicemap((:alpha, alpha)))
        return trace
    end

    # gibbs sampler for gaussain mean
    function mu_gibbs(trace, k)
        xNoise = trace[:Outcome=>:xNoise]
        muVar = trace[:Outcome=>:muVar]
        n = length(trace[:Cluster])
        X = [trace[:Outcome=>:X=>i=>:normal] for i in 1:n]
        class_label = [trace[:Cluster=>i=>:c] for i in 1:n]

        for c in 1:k
            nk = length(class_label[class_label.==c])
            Xk = mean(X[class_label.==c])

            # likelihood for new mu and variance
            mu = (nk/xNoise)/(nk/xNoise+1/muVar)*Xk
            lmda = 1/(nk/xNoise + 1/muVar)

            (trace, w, _, discard) = update(trace, get_args(trace), (),
                choicemap((:Outcome=>:mu=>c=>:normal, normal(mu, lmda))))
        end
        return trace
    end

    # gibbs sampler for class labels
    function class_gibbs(trace, k)
        n = length(trace[:Cluster])
        alpha = trace[:alpha] # current alpha
        class_label = [trace[:Cluster => i =>:c] for i in 1:n] # current assignment for all N
        X = [trace[:Outcome=>:X=>i=>:normal] for i in 1:n] # current X
        mu_dict = [trace[:Outcome=>:mu=>i=>:normal] for i in 1:k] # current mu
        xNoise = trace[:Outcome=>:xNoise]

        sample_per_cluster = Dict()
        for c in 1:k
            sample_per_cluster[c] = length(class_label.==c)
        end

        for i in 1:n
            c = class_label[i]
            x = X[i]
            sample_per_cluster[c] -= 1 # remove from sample

            p = fill(0.0, k) # build prob
            for i in 1:k
                mk = sample_per_cluster[i]
                pX = Gen.logpdf(normal, x, mu_dict[i], xNoise)
                p[i] = log(mk + alpha/k) - log(n-1+alpha) + pX
            end
            p = exp.(p .- (maximum(p)))
            p = p ./ (sum(p))

            c = categorical(p) # sample new obj
            sample_per_cluster[c] += 1
            class_label[i] = c
        end

        args = get_args(trace)
        cmap = []
        for i in 1:n
            push!(cmap, (:Cluster=>i=>:c, class_label[i]))
        end
        (trace, w, _, discard) = update(trace, args, (), choicemap(tuple(cmap...)...))
        return trace
    end

    export alpha_gibbs, mu_gibbs, class_gibbs
end
