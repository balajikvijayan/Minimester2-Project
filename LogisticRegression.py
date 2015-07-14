class LogisticRegression(BaseEstimator, LinearClassifierMixin,
                         _LearntSelectorMixin, SparseCoefMixin):
    """Logistic Regression (aka logit, MaxEnt) classifier.
    In the multiclass case, the training algorithm uses the one-vs-rest (OvR)
    scheme if the 'multi_class' option is set to 'ovr' and uses the
    cross-entropy loss, if the 'multi_class' option is set to 'multinomial'.
    (Currently the 'multinomial' option is supported only by the 'lbfgs' and
    'newton-cg' solvers.)
    This class implements regularized logistic regression using the
    `liblinear` library, newton-cg and lbfgs solvers. It can handle both
    dense and sparse input. Use C-ordered arrays or CSR matrices containing
    64-bit floats for optimal performance; any other input format will be
    converted (and copied).
    The newton-cg and lbfgs solvers support only L2 regularization with primal
    formulation. The liblinear solver supports both L1 and L2 regularization,
    with a dual formulation only for the L2 penalty.
    Parameters
    ----------
    penalty : str, 'l1' or 'l2'
        Used to specify the norm used in the penalization. The newton-cg and
        lbfgs solvers support only l2 penalties.
    dual : bool
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.
    C : float, optional (default=1.0)
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.
    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added the decision function.
    intercept_scaling : float, default: 1
        Useful only if solver is liblinear.
        when self.fit_intercept is True, instance vector x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equals to
        intercept_scaling is appended to the instance vector.
        The intercept becomes intercept_scaling * synthetic feature weight
        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.
    class_weight : {dict, 'auto'}, optional
        Over-/undersamples the samples of each class according to the given
        weights. If not given, all classes are supposed to have weight one.
        The 'auto' mode selects weights inversely proportional to class
        frequencies in the training set.
    max_iter : int
        Useful only for the newton-cg and lbfgs solvers. Maximum number of
        iterations taken for the solvers to converge.
    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use when
        shuffling the data.
    solver : {'newton-cg', 'lbfgs', 'liblinear'}
        Algorithm to use in the optimization problem.
    tol : float, optional
        Tolerance for stopping criteria.
    multi_class : str, {'ovr', 'multinomial'}
        Multiclass option can be either 'ovr' or 'multinomial'. If the option
        chosen is 'ovr', then a binary problem is fit for each label. Else
        the loss minimised is the multinomial loss fit across
        the entire probability distribution. Works only for the 'lbfgs'
        solver.
    verbose : int
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.
    Attributes
    ----------
    coef_ : array, shape (n_classes, n_features)
        Coefficient of the features in the decision function.
    intercept_ : array, shape (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.
        If `fit_intercept` is set to False, the intercept is set to zero.
    n_iter_ : int
        Maximum of the actual number of iterations across all classes.
        Valid only for the liblinear solver.
    See also
    --------
    SGDClassifier : incrementally trained logistic regression (when given
        the parameter ``loss="log"``).
    sklearn.svm.LinearSVC : learns SVM models using the same algorithm.
    Notes
    -----
    The underlying C implementation uses a random number generator to
    select features when fitting the model. It is thus not uncommon,
    to have slightly different results for the same input data. If
    that happens, try with a smaller tol parameter.
    Predict output may not match that of standalone liblinear in certain
    cases. See :ref:`differences from liblinear <liblinear_differences>`
    in the narrative documentation.
    References
    ----------
    LIBLINEAR -- A Library for Large Linear Classification
        http://www.csie.ntu.edu.tw/~cjlin/liblinear/
    Hsiang-Fu Yu, Fang-Lan Huang, Chih-Jen Lin (2011). Dual coordinate descent
        methods for logistic regression and maximum entropy models.
        Machine Learning 85(1-2):41-75.
        http://www.csie.ntu.edu.tw/~cjlin/papers/maxent_dual.pdf
    See also
    --------
    sklearn.linear_model.SGDClassifier
    """

    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0):

		#initializations
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector relative to X.
        Returns
        -------
        self : object
            Returns self.
        """
		#C must be positive
        if self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)"
                             % self.C)

		#Check X and y to have consistent length
        X, y = check_X_y(X, y, accept_sparse='csr', dtype=np.float64, order="C")
        #set the number of classes to be the number of unique y values
		self.classes_ = np.unique(y)
		#limits solvers to these types
        if self.solver not in ['liblinear', 'newton-cg', 'lbfgs']:
            raise ValueError(
                "Logistic Regression supports only liblinear, newton-cg and "
                "lbfgs solvers, Got solver=%s" % self.solver
                )
		#don't try these solvers for multinomial classification
        if self.solver == 'liblinear' and self.multi_class == 'multinomial':
            raise ValueError("Solver %s does not support a multinomial "
                             "backend." % self.solver)
		#limits multi_class options
        if self.multi_class not in ['ovr', 'multinomial']:
            raise ValueError("multi_class should be either ovr or multinomial "
                             "got %s" % self.multi_class)

		#if liblinear
        if self.solver == 'liblinear':
			#use liblinear algorithmn to set coef, intercept
            self.coef_, self.intercept_, self.n_iter_ = _fit_liblinear(
                X, y, self.C, self.fit_intercept, self.intercept_scaling,
                self.class_weight, self.penalty, self.dual, self.verbose,
                self.max_iter, self.tol
                )
			#done!
            return self
		#n classes
        n_classes = len(self.classes_)
		#classes
        classes_ = self.classes_
		#gotta have atleast 2
        if n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % classes_[0])
		#if 2 classes
        if len(self.classes_) == 2:
			#n is now 2
            n_classes = 1
			#classes self updates
            classes_ = classes_[1:]

		#coef are list
        self.coef_ = list()
		#intercept is as many zeros as classes
        self.intercept_ = np.zeros(n_classes)

        # Hack so that we iterate only once for the multinomial case.
		#^These folks are spooky good. If we have multinomials the classes are none
        if self.multi_class == 'multinomial':
            classes_ = [None]

		#loop through classes
        for ind, class_ in enumerate(classes_):
			#get our coefs
            coef_, _ = logistic_regression_path(
                X, y, pos_class=class_, Cs=[self.C],
                fit_intercept=self.fit_intercept, tol=self.tol,
                verbose=self.verbose, solver=self.solver,
                multi_class=self.multi_class, max_iter=self.max_iter,
                class_weight=self.class_weight)
            self.coef_.append(coef_[0])
		#squeeze some coefs
        self.coef_ = np.squeeze(self.coef_)
        # For the binary case, this get squeezed to a 1-D array.
		#^Yep.
		#if only 1 coef
        if self.coef_.ndim == 1:
			#set coef to the value
            self.coef_ = self.coef_[np.newaxis, :]
		#coef is it self as an array
        self.coef_ = np.asarray(self.coef_)
		#set intercept and coef
        if self.fit_intercept:
            self.intercept_ = self.coef_[:, -1]
            self.coef_ = self.coef_[:, :-1]

        return self

	def logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
								 max_iter=100, tol=1e-4, verbose=0,
								 solver='lbfgs', coef=None, copy=True,
								 class_weight=None, dual=False, penalty='l2',
								 intercept_scaling=1., multi_class='ovr'):
		"""Compute a Logistic Regression model for a list of regularization
		parameters.
		This is an implementation that uses the result of the previous model
		to speed up computations along the set of solutions, making it faster
		than sequentially calling LogisticRegression for the different parameters.
		Parameters
		----------
		X : array-like or sparse matrix, shape (n_samples, n_features)
			Input data.
		y : array-like, shape (n_samples,)
			Input data, target values.
		Cs : int | array-like, shape (n_cs,)
			List of values for the regularization parameter or integer specifying
			the number of regularization parameters that should be used. In this
			case, the parameters will be chosen in a logarithmic scale between
			1e-4 and 1e4.
		pos_class : int, None
			The class with respect to which we perform a one-vs-all fit.
			If None, then it is assumed that the given problem is binary.
		fit_intercept : bool
			Whether to fit an intercept for the model. In this case the shape of
			the returned array is (n_cs, n_features + 1).
		max_iter : int
			Maximum number of iterations for the solver.
		tol : float
			Stopping criterion. For the newton-cg and lbfgs solvers, the iteration
			will stop when ``max{|g_i | i = 1, ..., n} <= tol``
			where ``g_i`` is the i-th component of the gradient.
		verbose : int
			For the liblinear and lbfgs solvers set verbose to any positive
			number for verbosity.
		solver : {'lbfgs', 'newton-cg', 'liblinear'}
			Numerical solver to use.
		coef : array-like, shape (n_features,), default None
			Initialization value for coefficients of logistic regression.
		copy : bool, default True
			Whether or not to produce a copy of the data. Setting this to
			True will be useful in cases, when logistic_regression_path
			is called repeatedly with the same data, as y is modified
			along the path.
		class_weight : {dict, 'auto'}, optional
			Over-/undersamples the samples of each class according to the given
			weights. If not given, all classes are supposed to have weight one.
			The 'auto' mode selects weights inversely proportional to class
			frequencies in the training set.
		dual : bool
			Dual or primal formulation. Dual formulation is only implemented for
			l2 penalty with liblinear solver. Prefer dual=False when
			n_samples > n_features.
		penalty : str, 'l1' or 'l2'
			Used to specify the norm used in the penalization. The newton-cg and
			lbfgs solvers support only l2 penalties.
		intercept_scaling : float, default 1.
			This parameter is useful only when the solver 'liblinear' is used
			and self.fit_intercept is set to True. In this case, x becomes
			[x, self.intercept_scaling],
			i.e. a "synthetic" feature with constant value equals to
			intercept_scaling is appended to the instance vector.
			The intercept becomes intercept_scaling * synthetic feature weight
			Note! the synthetic feature weight is subject to l1/l2 regularization
			as all other features.
			To lessen the effect of regularization on synthetic feature weight
			(and therefore on the intercept) intercept_scaling has to be increased.
		multi_class : str, {'ovr', 'multinomial'}
			Multiclass option can be either 'ovr' or 'multinomial'. If the option
			chosen is 'ovr', then a binary problem is fit for each label. Else
			the loss minimised is the multinomial loss fit across
			the entire probability distribution. Works only for the 'lbfgs'
			solver.
		Returns
		-------
		coefs : ndarray, shape (n_cs, n_features) or (n_cs, n_features + 1)
			List of coefficients for the Logistic Regression model. If
			fit_intercept is set to True then the second dimension will be
			n_features + 1, where the last item represents the intercept.
		Cs : ndarray
			Grid of Cs used for cross-validation.
		Notes
		-----
		You might get slighly different results with the solver liblinear than
		with the others since this uses LIBLINEAR which penalizes the intercept.
		"""
		#Cs set to be in log space from -4 to 4
		if isinstance(Cs, numbers.Integral):
			Cs = np.logspace(-4, 4, Cs)
		#multi_class check
		if multi_class not in ['multinomial', 'ovr']:
			raise ValueError("multi_class can be either 'multinomial' or 'ovr'"
							 "got %s" % multi_class)
		#solver check
		if solver not in ['liblinear', 'newton-cg', 'lbfgs']:
			raise ValueError("Logistic Regression supports only liblinear,"
							 " newton-cg and lbfgs solvers. got %s" % solver)
		#multiclass and solver check
		if multi_class == 'multinomial' and solver == 'liblinear':
			raise ValueError("Solver %s cannot solve problems with "
							 "a multinomial backend." % solver)
		#newtog-cg and lbfgs solver check
		if solver != 'liblinear':
			if penalty != 'l2':
				raise ValueError("newton-cg and lbfgs solvers support only "
								 "l2 penalties, got %s penalty." % penalty)
			if dual:
				raise ValueError("newton-cg and lbfgs solvers support only "
								 "dual=False, got dual=%s" % dual)
		# Preprocessing.
		#^Check X,y
		X = check_array(X, accept_sparse='csr', dtype=np.float64)
		y = check_array(y, ensure_2d=False, copy=copy, dtype=None)
		#odd notation trick
		_, n_features = X.shape
		#check, again.
		check_consistent_length(X, y)
		#set classes = uniques in y
		classes = np.unique(y)
		#multionmial check
		if pos_class is None and multi_class != 'multinomial':
			if (classes.size > 2):
				raise ValueError('To fit OvR, use the pos_class argument')
			# np.unique(y) gives labels in sorted order.
			pos_class = classes[1]

		# If class_weights is a dict (provided by the user), the weights
		# are assigned to the original labels. If it is "auto", then
		# the class_weights are assigned after masking the labels with a OvR.
		sample_weight = np.ones(X.shape[0])
		le = LabelEncoder()
		#if class weight is a dict
		if isinstance(class_weight, dict):
			#if liblinear
			if solver == "liblinear":
				if classes.size == 2:
					# Reconstruct the weights with keys 1 and -1
					temp = {1: class_weight[pos_class],
							-1: class_weight[classes[0]]}
					class_weight = temp.copy()
				#can't do more than 2 classes with dictionary
				else:
					raise ValueError("In LogisticRegressionCV the liblinear "
									 "solver cannot handle multiclass with "
									 "class_weight of type dict. Use the lbfgs, "
									 "newton-cg solvers or set "
									 "class_weight='auto'")
			#get class weight and sample_weight
			else:
				class_weight_ = compute_class_weight(class_weight, classes, y)
				sample_weight = class_weight_[le.fit_transform(y)]

		# For doing a ovr, we need to mask the labels first. for the
		# multinomial case this is not necessary.
		if multi_class == 'ovr':
			#w0 is an array of zeros that's the size of n features + the closest int for the fit intercept
			w0 = np.zeros(n_features + int(fit_intercept))
			#mask classes for positive and negative
			mask_classes = [-1, 1]
			mask = (y == pos_class)
			y[mask] = 1
			y[~mask] = -1
			# To take care of object dtypes, i.e 1 and -1 are in the form of
			# strings.
			y = as_float_array(y, copy=False)

		#this is for the none ovr multi_class
		else:
			#binarize (0 or 1) some labels
			lbin = LabelBinarizer()
			Y_bin = lbin.fit_transform(y)
			if Y_bin.shape[1] == 1:
				Y_bin = np.hstack([1 - Y_bin, Y_bin])
			w0 = np.zeros((Y_bin.shape[1], n_features + int(fit_intercept)),
						  order='F')
			mask_classes = classes
		#class weighting
		if class_weight == "auto":
			class_weight_ = compute_class_weight(class_weight, mask_classes, y)
			sample_weight = class_weight_[le.fit_transform(y)]
		#if the coef is not None
		if coef is not None:
			# it must work both giving the bias term and not
			#specific for multi_class = 'ovr
			if multi_class == 'ovr':
				#coef size constraints
				if coef.size not in (n_features, w0.size):
					raise ValueError(
						'Initialization coef is of shape %d, expected shape '
						'%d or %d' % (coef.size, n_features, w0.size)
						)
				w0[:coef.size] = coef
			#other multi_class
			else:
				# For binary problems coef.shape[0] should be 1, otherwise it
				# should be classes.size.
				#setup n_vectors
				n_vectors = classes.size
				if n_vectors == 2:
					n_vectors = 1
				#initialization check
				if (coef.shape[0] != n_vectors or
						coef.shape[1] not in (n_features, n_features + 1)):
					raise ValueError(
						'Initialization coef is of shape (%d, %d), expected '
						'shape (%d, %d) or (%d, %d)' % (
							coef.shape[0], coef.shape[1], classes.size,
							n_features, classes.size, n_features + 1
							)
						)
				w0[:, :coef.shape[1]] = coef
		#multinomial check
		if multi_class == 'multinomial':
			# fmin_l_bfgs_b and newton-cg accepts only ravelled parameters.
			#ravel (flatten) the w0 array
			w0 = w0.ravel()
			#setup a target
			target = Y_bin
			if solver == 'lbfgs':
				#do multinomial gradient loss on X
				func = lambda x, *args: _multinomial_loss_grad(x, *args)[0:2]
			elif solver == 'newton-cg':
				#do multinomial loss, multinomial gradient loss and multinomail loss grad hess on X
				func = lambda x, *args: _multinomial_loss(x, *args)[0]
				grad = lambda x, *args: _multinomial_loss_grad(x, *args)[1]
				hess = _multinomial_loss_grad_hess
		#binomial class
		else:
			#setup target
			target = y
			if solver == 'lbfgs':
				#logistic loss and grad function
				func = _logistic_loss_and_grad
			elif solver == 'newton-cg':
				#func, grad and hess for X (logistic loss instead of multinomial loss)
				func = _logistic_loss
				grad = lambda x, *args: _logistic_loss_and_grad(x, *args)[1]
				hess = _logistic_loss_grad_hess
		#coefs in a list
		coefs = list()
		#loop through all the Cs
		for C in Cs:
			#lbfgs solver
			if solver == 'lbfgs':
				#we'll try finding min 1 bfgs b
				try:
					w0, loss, info = optimize.fmin_l_bfgs_b(
						func, w0, fprime=None,
						args=(X, target, 1. / C, sample_weight),
						iprint=(verbose > 0) - 1, pgtol=tol, maxiter=max_iter
						)
				except TypeError:
					# old scipy doesn't have maxiter
					#try finding min 1 bfgs b without a max iteration limit
					w0, loss, info = optimize.fmin_l_bfgs_b(
						func, w0, fprime=None,
						args=(X, target, 1. / C, sample_weight),
						iprint=(verbose > 0) - 1, pgtol=tol
						)
				#gotta have enough iterations to converge
				if info["warnflag"] == 1 and verbose > 0:
					warnings.warn("lbfgs failed to converge. Increase the number "
								  "of iterations.")
			#newton-cg solver
			elif solver == 'newton-cg':
				#set args and w0
				args = (X, target, 1. / C, sample_weight)
				w0 = newton_cg(hess, func, grad, w0, args=args, maxiter=max_iter,
							   tol=tol)
			#liblinear solver
			elif solver == 'liblinear':
				#get coef and intercept from fit_liblinear
				coef_, intercept_, _, = _fit_liblinear(
					X, y, C, fit_intercept, intercept_scaling, class_weight,
					penalty, dual, verbose, max_iter, tol,
					)
				#get fit_intercept
				if fit_intercept:
					w0 = np.concatenate([coef_.ravel(), intercept_])
				#get w0
				else:
					w0 = coef_.ravel()
			else:
				raise ValueError("solver must be one of {'liblinear', 'lbfgs', "
								 "'newton-cg'}, got '%s' instead" % solver)
			#multinomial
			if multi_class == 'multinomial':
				#get w0
				multi_w0 = np.reshape(w0, (classes.size, -1))
				#2 classes
				if classes.size == 2:
					multi_w0 = multi_w0[1][np.newaxis, :]
				#setup coefs
				coefs.append(multi_w0)
			else:
				coefs.append(w0)
		return coefs, np.array(Cs)
	
#So this uses external classes (liblinear), (newton-cg) and (lbfgs) solvers in order to implement regularized regression. Liblinear is a C library for large linear classification with support for L1 and L2 regularization; its highly performant due to being written in C. Newton-cg and lbfgs solvers are optimization methods: see here for more information: http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/.
