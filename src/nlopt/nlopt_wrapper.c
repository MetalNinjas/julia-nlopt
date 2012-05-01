// File: nlopt_wrapper.c
// Author: Jannis Harder
// Description: Wrapper for nlopt using context switching

#include <stdlib.h>
#include <ucontext.h>
#include <string.h>
#include <stdio.h> // For debugging TODO: remove
#include <nlopt.h>

#define STACKSIZE (8 * 1024 * 1024)

enum status {
	ST_IDLE = 0,
	ST_RUNNING,
	ST_VALUE,
	ST_GRAD,
	ST_DONE
};

struct function {
	struct wrapper * w;
	struct function * next;
	int id;
};

struct wrapper {
	struct function f;
	nlopt_opt opt;
	int n;
	ucontext_t julia_ctx;
	ucontext_t nlopt_ctx;
	char *nlopt_stack;
	double final_f;
	double *opt_x;
	double *eval_x;
	double *eval_grad;
	double eval_f;
	int force_stop;
	int result;
	int function_id;
	enum status status;
};


double nlopt_wrapper_f(unsigned n, const double *x, double *grad, void *f_data);
void nlopt_wrapper_optimize_thread(struct wrapper* w);

void nlopt_wrapper_version(int *version)
{
	nlopt_version(&version[0], &version[1], &version[2]);
	version[3] = 2;
	version[4] = 2;
	version[5] = 4;
	version[6] = 0;
}

struct wrapper *nlopt_wrapper_create(int type_id, int dimensions)
{
	nlopt_opt opt = nlopt_create(type_id, dimensions);
	if (!opt)
		return NULL;
	struct wrapper *w = calloc(1, sizeof(struct wrapper));
	w->n = dimensions;
	w->opt = opt;
	w->f.w = w;
	w->f.id = 1;
	return w;
}

void nlopt_wrapper_free(struct wrapper *w)
{
	nlopt_destroy(w->opt);
	free(w->opt_x);
	struct function *f = &w->f;
	while (f) {
		struct function *next = f->next;
		free(f);
		f = next;
	}
}

int nlopt_wrapper_objective(struct wrapper *w, int max)
{
	if (max)
		return nlopt_set_max_objective(w->opt, &nlopt_wrapper_f, &w->f);
	else
		return nlopt_set_min_objective(w->opt, &nlopt_wrapper_f, &w->f);
}


double nlopt_wrapper_f(unsigned n, const double *x, double *grad, void *f_data)
{
	struct function *f = f_data;
	struct wrapper *w = f->w;

	w->function_id = f->id;

	w->status = grad ? ST_GRAD : ST_VALUE;
	memcpy(w->eval_x, x, n * sizeof(double));
	swapcontext(&w->nlopt_ctx, &w->julia_ctx);
	if (w->force_stop)
		nlopt_force_stop(w->opt);
	if (grad)
		memcpy(grad, w->eval_grad, n * sizeof(double));
	return w->eval_f;
}

void nlopt_wrapper_optimize_start(struct wrapper *w, double *x)
{
	int i;
	getcontext(&w->nlopt_ctx);
	w->nlopt_stack = calloc(1, STACKSIZE);
	w->nlopt_ctx.uc_stack.ss_sp = w->nlopt_stack;
	w->nlopt_ctx.uc_stack.ss_size = STACKSIZE;
	w->nlopt_ctx.uc_link = &w->julia_ctx;
	makecontext(&w->nlopt_ctx, (void(*)())&nlopt_wrapper_optimize_thread, 1, w);


	w->opt_x = calloc(w->n, sizeof(double));
	memcpy(w->opt_x, x, w->n * sizeof(double));
}

int nlopt_wrapper_optimize_callback(struct wrapper *w, double *x, double *grad, double f, int force_stop, int *function_id)
{
	w->eval_f = f;
	w->eval_x = x;
	w->eval_grad = grad;
	w->force_stop = force_stop;
	swapcontext(&w->julia_ctx, &w->nlopt_ctx);
	*function_id = w->function_id;
	return w->status;
}

int nlopt_wrapper_optimize_finalize(struct wrapper *w, double *x, double *f)
{
	memcpy(x, w->opt_x, w->n * sizeof(double));
	free(w->opt_x);
	w->opt_x = NULL;
	*f = w->final_f;
	free(w->nlopt_stack);
	w->nlopt_stack = NULL;
	return w->result;
}

void nlopt_wrapper_optimize_thread(struct wrapper *w)
{
	w->result = nlopt_optimize(w->opt, w->opt_x, &w->final_f);
	w->status = ST_DONE;
}

void nlopt_wrapper_dimopt(struct wrapper *w, double *x, int i)
{
	switch (i) {
	case 0: nlopt_set_lower_bounds(w->opt, x); break;
	case 1: nlopt_set_upper_bounds(w->opt, x); break;
	case 2: nlopt_set_xtol_abs(w->opt, x); break;
	}
}

void nlopt_wrapper_doubleopt(struct wrapper *w, double v, int i)
{
	switch (i) {
	case 0: nlopt_set_stopval(w->opt, v); break;
	case 1: nlopt_set_ftol_rel(w->opt, v); break;
	case 2: nlopt_set_ftol_abs(w->opt, v); break;
	case 3: nlopt_set_xtol_rel(w->opt, v); break;
	case 4: nlopt_set_maxtime(w->opt, v); break;
	}
}

void nlopt_wrapper_intopt(struct wrapper *w, long int v, int i)
{
	switch (i) {
	case 0: nlopt_set_maxeval(w->opt, v); break;
	case 1: nlopt_set_population(w->opt, v < 0 ? 0 : v); break;
	case 2: nlopt_srand(v); break;
	}
}

void nlopt_wrapper_add_constraint(struct wrapper *w, int id, double tolerance, int equality)
{
	struct function *f = calloc(1, sizeof(struct function));

	f->w = w;
	f->next = w->f.next;
	f->id = id;
	w->f.next = f;
	if (equality)
		nlopt_add_equality_constraint(w->opt, &nlopt_wrapper_f, f, tolerance);
	else
		nlopt_add_inequality_constraint(w->opt, &nlopt_wrapper_f, f, tolerance);
}

void nlopt_wrapper_local_optimizer(struct wrapper *w, struct wrapper *local_w)
{
	nlopt_set_local_optimizer(w->opt, local_w->opt);
}
