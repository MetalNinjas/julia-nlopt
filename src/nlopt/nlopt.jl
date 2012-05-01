# File: nlopt.jl
# Author: Jannis Harder
# Description: Bindings to NLopt

let
    Option(T...) = Union((), T...)

    global NLoptConstraint
    type NLoptConstraint
        f::Function
        tolerance::Float64
        equality::Bool
    end

    global nlopt_equality
    function nlopt_equality(f, t)
        NLoptConstraint(f, t, true)
    end

    global nlopt_inequality
    function nlopt_inequality(f, t)
        NLoptConstraint(f, t, false)
    end
    nlopt_inequality(f) = nlopt_inequality(f, 0.0)

    global NLopt
    type NLopt
        algorithm::Option(Symbol)
        min_objective::Option(Function) # set only one of these two
        max_objective::Option(Function)
        lower_bounds::Option(Array{Float64,1}, Float64)
        upper_bounds::Option(Array{Float64,1}, Float64)
        constraints::Array{NLoptConstraint, 1}
        stopval::Option(Float64)
        ftol_rel::Option(Float64)
        ftol_abs::Option(Float64)
        xtol_rel::Option(Float64)
        xtol_abs::Option(Array{Float64,1}, Float64)
        maxeval::Option(Int)
        maxtime::Option(Float64)
        local_optimizer::Option(NLopt)
        initial_step::Option(Array{Float64,1}, Float64)
        population::Option(Int)
        srand::Option(Int)

        function NLopt(algorithm)
            new(algorithm, #algorithm
                (), (), (), (),
                Array(NLoptConstraint, 0), # constraint lists
                (), (), (), (), (), (), (), (), (), (), ())
        end
    end

    NLopt() = NLopt(())
    function NLopt(algorithm, args...)
        opt = NLopt(algorithm)
        if mod(length(args), 2) == 1
            if !isa(args[end], Symbol)
                error("nlopt: expected argument name")
            end
            error("nlopt: no value given for parameter ", args[end])
        end
        for i in 1:div(length(args), 2)
            sym = args[2*i-1]
            if !isa(sym, Symbol)
                error("nlopt: expected argument name")
            end
            expr = Expr(:., {:opt, Expr(:quote, {sym}, Any)}, Any)
            eval(:((opt, v) -> $expr = v))(opt, convert(fieldtype(opt, sym),args[2*i]))
        end
        opt
    end

    global nlopt_wrapper_version = (2,2,4,0)

    global nlopt_algorithms =
    [:gn_direct,
     :gn_direct_l,
     :gn_direct_l_rand,
     :gn_direct_noscal,
     :gn_direct_l_noscal,
     :gn_direct_l_rand_noscal,
     :gn_orig_direct,
     :gn_orig_direct_l,
     :gd_stogo,
     :gd_stogo_rand,
     :ld_lbfgs_nocedal,
     :ld_lbfgs,
     :ln_praxis,
     :ld_var1,
     :ld_var2,
     :ld_tnewton,
     :ld_tnewton_restart,
     :ld_tnewton_precond,
     :ld_tnewton_precond_restart,
     :gn_crs2_lm,
     :gn_mlsl,
     :gd_mlsl,
     :gn_mlsl_lds,
     :gd_mlsl_lds,
     :ld_mma,
     :ln_cobyla,
     :ln_newuoa,
     :ln_newuoa_bound,
     :ln_neldermead,
     :ln_sbplx,
     :ln_auglag,
     :ld_auglag,
     :ln_auglag_eq,
     :ld_auglag_eq,
     :ln_bobyqa,
     :gn_isres,
     :auglag,
     :auglag_eq,
     :g_mlsl,
     :g_mlsl_lds,
     :ld_slsqp]

    algorithm_map = HashTable()

    for i in 1:length(nlopt_algorithms)
        algorithm_map[nlopt_algorithms[i]] = i - 1
    end

    # makes wrapping of c code simpler
    function wrap_dlopen(name, prefix)
        dl = dlopen(name)
        function wrap(sym, ret, types...)
            symptr = dlsym(dl, symbol(strcat(prefix, sym)))
            nargs = length(types)
            nsyms = map(i -> symbol(strcat("a", i)), [1:length(types)])
            eval(:(($nsyms...,) -> ccall($symptr, $ret, ($types...,), $nsyms...)))
        end
        return wrap
    end

    isnull(x) = int(x) == 0
    isset(x) = () != x

    run(`make -C $(require_resource("")) -s`)
    wrapper_dl = wrap_dlopen(require_resource("nlopt_wrapper.so"), :nlopt_wrapper_)

    version = Array(Int32, 7)
    wrapper_dl(:version, Void, Ptr{Int32})(version)

    nlopt_wrapper_lib_version = tuple(int(version[4:])...)
    global nlopt_lib_version = tuple(int(version[1:3])...)

    if nlopt_wrapper_lib_version != nlopt_wrapper_version
        error("nlopt: mismatch between wrapper version ",
              nlopt_wrapper_version, " and wrapper lib version ",
              nlopt_wrapper_lib_version)
    end
    if nlopt_lib_version != nlopt_wrapper_version[1:3]
        println("Warning: nlopt: mismatch between wrapper version ",
                nlopt_wrapper_version, " and nlopt version ",
                nlopt_lib_version)
    end

    create = wrapper_dl(:create, Ptr{Void}, Int32, Int32)
    free = wrapper_dl(:free, Void, Ptr{Void})
    objective = wrapper_dl(:objective, Int32, Ptr{Void}, Int32)
    optimize_start = wrapper_dl(:optimize_start, Int32, Ptr{Void}, Ptr{Float64})
    optimize_callback = wrapper_dl(:optimize_callback, Int32, Ptr{Void}, Ptr{Float64}, Ptr{Float64}, Float64, Int32, Ptr{Int32})
    optimize_finalize = wrapper_dl(:optimize_finalize, Int32, Ptr{Void}, Ptr{Float64}, Ptr{Float64})
    dimopt = wrapper_dl(:dimopt, Void, Ptr{Void}, Ptr{Float64}, Int32)
    doubleopt = wrapper_dl(:doubleopt, Void, Ptr{Void}, Float64, Int32)
    intopt = wrapper_dl(:intopt, Void, Ptr{Void}, Int, Int32)
    add_constraint = wrapper_dl(:add_constraint, Void, Ptr{Void}, Int32, Float64, Int32)
    local_optimizer = wrapper_dl(:local_optimizer, Void, Ptr{Void}, Ptr{Void})

    const st_idle = 0
    const st_running = 1
    const st_value = 2
    const st_grad = 3
    const st_done = 4

    function ensure_width(n, x)
        if isa(x, Float64)
            return (0 * [1:n]) + x
        elseif length(x) != n
            error("nlopt: dimension mismatch (got ", length(x), ", expected ", n)
        else
            return x
        end
    end


    function set_options(w, opt, n)
        if isset(opt.lower_bounds)
            dimopt(w, ensure_width(n, opt.lower_bounds), 0)
        end
        if isset(opt.upper_bounds)
            dimopt(w, ensure_width(n, opt.upper_bounds), 1)
        end
        if isset(opt.xtol_abs)
            dimopt(w, ensure_width(n, opt.xtol_abs), 1)
        end
        if isset(opt.stopval)
            doubleopt(w, opt.stopval, 0)
        end
        if isset(opt.ftol_rel)
            doubleopt(w, opt.ftol_rel, 1)
        end
        if isset(opt.ftol_abs)
            doubleopt(w, opt.ftol_abs, 2)
        end
        if isset(opt.xtol_rel)
            doubleopt(w, opt.xtol_rel, 3)
        end
        if isset(opt.maxtime)
            doubleopt(w, opt.maxtime, 4)
        end
        if isset(opt.maxeval)
            intopt(w, opt.maxeval, 0)
        end
        if isset(opt.population)
            intopt(w, opt.population, 1)
        end
        if isset(opt.srand)
            intopt(w, opt.srand, 2)
        end
    end

    function create_handle(opt, n)
        if opt.algorithm == ()
            error("nlopt: no algorithm specified")
        elseif !has(algorithm_map, opt.algorithm)
            error("nlopt: unknown algorithm ", opt.algorithm)
        end
        algorithm_id = algorithm_map[opt.algorithm]
        w = create(algorithm_id, n)
        if isnull(w)
            error("nlopt: could not initialize")
        end
        return w
    end

    function ignorable_arg(f)
        wrapped = false
        function f2(x, d)
            return f(x)
        end
        function f1(x, d)
            local r
            try
                r = f(x, d)
                f1 = f
            catch err
                f1 = f2
                r = f1(x, d)
            end
            return r
        end
        return (x, d) -> f1(x, d)
    end

    global nlopt_optimize
    function nlopt_optimize(opt, x::Array{Float64, 1})
        n = length(x)
        w = create_handle(opt, n)
        ret = ()
        try
            if isset(opt.max_objective) && isset(opt.min_objective)
                error("nlopt: only one of max_objective and min_objective can be used")
            elseif isset(opt.max_objective)
                objective(w, 1)
                objective_f = opt.max_objective
            elseif isset(opt.min_objective)
                objective(w, 0)
                objective_f = opt.min_objective
            else
                error("nlopt: no objective function specified")
            end

            set_options(w, opt, n)

            if isset(opt.local_optimizer)
                local_w = create_handle(opt.local_optimizer, n)
                try
                    set_options(local_w, opt.local_optimizer, n)
                catch err
                    free(local_w)
                    throw(err)
                end
                local_optimizer(w, local_w)
                free(local_w)
            end

            functions = Array(Function, 1)
            functions[1] = ignorable_arg(objective_f)

            for c in opt.constraints
                push(functions, ignorable_arg(c.f))
                add_constraint(w, length(functions), c.tolerance, c.equality ? 1 : 0)
            end

            optimize_start(w, x)
            x = Array(Float64, n)
            grad = Array(Float64, n)
            f = 0.0
            function_id = Array(Int32, 1)
            function_id[1] = 1
            exception = ()
            wrapped = false
            force_stop = 0
            while true
                status = optimize_callback(w, x, grad, f, force_stop, function_id)
                if status == st_done
                    break
                elseif status == st_value || status == st_grad
                    is_grad = status == st_grad
                    r = 0.0
                    try
                        r = functions[function_id[1]](x, is_grad)
                    catch err
                        exception = err
                        force_stop = 1
                        continue
                    end
                    if is_grad
                        (f, grad) = r
                    else
                        if isa(r, Tuple)
                            f = r[1]
                        else
                            f = r
                        end
                    end
                else
                    assert(false, "invalid state")
                end
            end
            f_ptr = Array(Float64, 1)
            res_code = optimize_finalize(w, x, f_ptr)
            res = {-1 => :failure,
                   -2 => :invalid_args,
                   -3 => :out_of_memory,
                   -4 => :roundoff_limited,
                   -5 => :forced_stop,
                   1 => :success,
                   2 => :stopval_reached,
                   3 => :ftol_reached,
                   4 => :xtol_reached,
                   5 => :maxeval_reached,
                   6 => :maxtime_reached}[res_code]
            if res == :forced_stop
                if isa(exception, Exception)
                    throw(exception)
                end
                res = exception
            elseif res_code < 0
                error("nlopt: optimization failed, reason: ", res)
            end
            ret = (res, x, f_ptr[1])
        catch x
            free(w)
            throw(x)
        end
        free(w)
        return ret
    end
end
