import tensorflow as tf
import numpy as np
import matplotlib.pyplot as PLT
import tflowtools as TFT
import random


# ******* A General Artificial Neural Network ********
# This is the original GANN, which has been improved in the file gann.py

class Gann():

    def __init__(self, dims, cman, lrate=.1, mbs=10, vint=None, softmax=True, optimize_func="adam", error="crossentropy"):
        self.learning_rate = lrate
        self.layer_sizes = dims  # Sizes of each layer of neurons
        self.grabvars = []  # Variables to be monitored (by gann code) during a run.
        self.dendrogram_vars = []
        self.minibatch_size = mbs
        self.validation_interval = vint
        self.validation_history = []
        self.caseman = cman
        self.softmax_outputs = softmax
        self.modules = []
        self.build(optimize_func, error)

    # Probed variables are to be displayed in the Tensorboard.
    def gen_probe(self, module_index, type, spec):
        self.modules[module_index].gen_probe(type, spec)

    # Grabvars are displayed by my own code, so I have more control over the display format.  Each
    # grabvar gets its own matplotlib figure in which to display its value.
    def add_grabvar(self, module_index, type='wgt'):
        self.grabvars.append(self.modules[module_index].getvar(type))

    def add_dendrogram_var(self, module_index, type='wgt'):
        self.dendrogram_vars.append(self.modules[module_index].getvar(type))

    def roundup_probes(self):
        self.probes = tf.summary.merge_all()

    def add_module(self, module):
        self.modules.append(module)

    def build(self, optimizer, error):
        tf.reset_default_graph()  # This is essential for doing multiple runs!!
        num_inputs = self.layer_sizes[0]
        self.input = tf.placeholder(tf.float64, shape=(None, num_inputs), name='Input')
        invar = self.input
        insize = num_inputs
        # Build all of the modules
        for i, outsize in enumerate(self.layer_sizes[1:]):
            gmod = Gannmodule(self, i, invar, insize, outsize)
            invar = gmod.output
            insize = gmod.outsize
        self.output = gmod.output  # Output of last module is output of whole network
        if self.softmax_outputs:
            self.raw_output = self.output
            self.output = tf.nn.softmax(self.output)
        self.target = tf.placeholder(tf.float64, shape=(None, gmod.outsize), name='Target')
        self.configure_learning(optimizer, error)

    # The optimizer knows to gather up all "trainable" variables in the function graph and compute
    # derivatives of the error function with respect to each component of each variable, i.e. each weight
    # of the weight array.

    def configure_learning(self, optimize_func, error):
        self.predictor = self.output  # Simple prediction runs will request the value of output neurons

        # Defining the training operator
        if optimize_func.lower() == "graddescent":
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif optimize_func.lower() == "adam":
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif optimize_func.lower() == "rms":
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif optimize_func.lower() == "adagrad":
            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        else:
            optimizer = None

        # Defining the loss function
        if error.lower() == "mse":
            self.error = tf.reduce_mean(tf.square(self.target - self.output), name='MSE')
        elif error.lower() == "crossentropy":
            self.error = tf.losses.softmax_cross_entropy(self.target, self.raw_output)

        self.trainer = optimizer.minimize(self.error, name='Backprop')

    def do_training(self, sess, cases, steps=100, continued=False):
        if not continued:
            self.error_history = []
        for step in range(steps):
            gvars = [self.error] + self.grabvars
            mbs = self.minibatch_size
            minibatch = []
            while len(minibatch) < mbs:
                number = random.randint(0, len(cases)-1)
                minibatch.append(cases[number])
            inputs = [c[0] for c in minibatch]
            targets = [c[1] for c in minibatch]
            feeder = {self.input: inputs, self.target: targets}
            _, grabvals, _ = self.run_one_step([self.trainer], gvars, self.probes, session=sess,
                                               feed_dict=feeder, step=step)
            self.error_history.append((step, grabvals[0]))
            self.consider_validation_testing(step, sess)
        TFT.plot_training_history(self.error_history, self.validation_history,
                                  xtitle="Steps", ytitle="Error", title="", fig=not continued)

    # bestk = 1 when you're doing a classification task and the targets are one-hot vectors.  This will invoke the
    # gen_match_counter error function. Otherwise, when
    # bestk=None, the standard MSE error function is used for testing.

    def do_testing(self, sess, cases, msg='Testing', bestk=None):
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        self.test_func = self.error
        if bestk is not None:
            self.test_func = self.gen_match_counter(self.predictor, [TFT.one_hot_to_int(list(v)) for v in targets],
                                                    k=bestk)
        testres, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, self.probes, session=sess,
                                                 feed_dict=feeder)
        if bestk is None:
            print('%s Set Error = %f ' % (msg, testres))
        else:
            print('%s Set Correct Classifications = %f %%' % (msg, 100 * (testres / len(cases))))
        return testres  # self.error uses MSE, so this is a per-case value when bestk=None

    def do_mapping(self, no_of_cases, defined_grab_vars=[], dendrogram_layers=[], bestk=None, dendrogram=False, labels=False):
        self.reopen_current_session()
        if len(defined_grab_vars) > 0:
            for (layer, type) in defined_grab_vars:
                if (layer-1, 'in') not in defined_grab_vars:
                    self.add_grabvar(layer-1, type)
        if len(dendrogram_layers) > 0:
            for layer in dendrogram_layers:
                self.add_dendrogram_var(layer-1, 'out')
        cases = self.get_random_cases(no_of_cases)
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        if bestk is not None:
            self.test_func = self.gen_match_counter(self.predictor, [TFT.one_hot_to_int(list(v)) for v in targets],
                                                    k=bestk)
        feeder = {self.input: inputs, self.target: targets}
        testres, grabvals, _ = self.run_one_step(self.test_func, [self.grabvars, self.dendrogram_vars], self.probes, session=self.current_session,
                                                 feed_dict=feeder)
        self.display_grabvars(grabvals[0], self.grabvars)
        if dendrogram:
            if labels:
                input_den = [target for target in targets]
            else:
                input_den = [TFT.bits_to_str(case) for case in inputs]
            for i, grabval in enumerate(grabvals[1]):
                TFT.dendrogram(grabval, input_den, title=self.dendrogram_vars[i].name)

    def get_random_cases(self, no_of_cases):
        total_cases = self.caseman.get_training_cases()
        cases = []
        while len(cases) < no_of_cases:
            rand = random.randint(0, len(total_cases)-1)
            cases.append(total_cases[rand])
        return cases

    # Logits = tensor, float - [batch_size, NUM_CLASSES].
    # labels: Labels tensor, int32 - [batch_size], with values in range [0, NUM_CLASSES).
    # in_top_k checks whether correct val is in the top k logit outputs.  It returns a vector of shape [batch_size]
    # This returns an OPERATION object that still needs to be RUN to get a count.
    # tf.nn.top_k differs from tf.nn.in_top_k in the way they handle ties.  The former takes the lowest index, while
    # the latter includes them ALL in the "top_k", even if that means having more than k "winners".  This causes
    # problems when ALL outputs are the same value, such as 0, since in_top_k would then signal a match for any
    # target.  Unfortunately, top_k requires a different set of arguments...and is harder to use.

    def gen_match_counter(self, logits, labels, k=1):
        correct = tf.nn.in_top_k(tf.cast(logits, tf.float32), labels, k)  # Return number of correct outputs
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def training_session(self, steps, sess=None, dir="probeview", continued=False):
        session = sess if sess else TFT.gen_initialized_session(dir=dir)
        self.current_session = session
        self.roundup_probes()  # this call must come AFTER the session is created, else graph is not in tensorboard.
        self.do_training(session, self.caseman.get_training_cases(), steps, continued=continued)

    def testing_session(self, sess, bestk=None):
        cases = self.caseman.get_testing_cases()
        if len(cases) > 0:
            self.do_testing(sess, cases, msg='Final Testing', bestk=bestk)

    def consider_validation_testing(self, epoch, sess):
        if self.validation_interval and (epoch % self.validation_interval == 0):
            cases = self.caseman.get_validation_cases()
            if len(cases) > 0:
                error = self.do_testing(sess, cases, msg='Validation Testing')
                self.validation_history.append((epoch, error))

    # Do testing (i.e. calc error without learning) on the training set.
    def test_on_trains(self, sess, bestk=None):
        self.do_testing(sess, self.caseman.get_training_cases(), msg='Total Training', bestk=bestk)

    # Similar to the "quickrun" functions used earlier.

    def run_one_step(self, operators, grabbed_vars=None, probed_vars=None, dir='probeview',
                     session=None, feed_dict=None, step=1):
        sess = session if session else TFT.gen_initialized_session(dir=dir)
        if probed_vars is not None:
            results = sess.run([operators, grabbed_vars, probed_vars], feed_dict=feed_dict)
            sess.probe_stream.add_summary(results[2], global_step=step)
        else:
            results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
        return results[0], results[1], sess

    def display_grabvars(self, grabbed_vals, grabbed_vars, step=1):
        names = [x.name for x in grabbed_vars]
        msg = "Grabbed Variables at Step " + str(step)
        print("\n" + msg, end="\n")
        for i, v in enumerate(grabbed_vals):
            if names:
                print("   " + names[i] + " = ", end="\n")
            if type(v) == np.ndarray and len(v.shape) > 1:  # If v is a matrix, use hinton plotting
                TFT.hinton_plot(v, title=names[i] + ' at step ' + str(step))
                TFT.display_matrix(v, title=names[i] + ' at step ' + str(step))
            else:
                print(v, end="\n\n")

    def run(self, steps=100, sess=None, continued=False, bestk=None):
        PLT.ion()
        self.training_session(steps, sess=sess, continued=continued)
        self.test_on_trains(sess=self.current_session, bestk=bestk)
        self.testing_session(sess=self.current_session, bestk=bestk)
        self.close_current_session(view=False)
        PLT.show()
        PLT.ioff()

    # After a run is complete, runmore allows us to do additional training on the network, picking up where we
    # left off after the last call to run (or runmore).  Use of the "continued" parameter (along with
    # global_training_step) allows easy updating of the error graph to account for the additional run(s).

    def runmore(self, epochs=100, bestk=None):
        self.reopen_current_session()
        self.run(epochs, sess=self.current_session, continued=True, bestk=bestk)

    #   ******* Saving GANN Parameters (weights and biases) *******************
    # This is useful when you want to use "runmore" to do additional training on a network.
    # spath should have at least one directory (e.g. netsaver), which you will need to create ahead of time.
    # This is also useful for situations where you want to first train the network, then save its parameters
    # (i.e. weights and biases), and then run the trained network on a set of test cases where you may choose to
    # monitor the network's activity (via grabvars, probes, etc) in a different way than you monitored during
    # training.

    def save_session_params(self, spath='netsaver/my_saved_session', sess=None, step=0):
        session = sess if sess else self.current_session
        state_vars = []
        for m in self.modules:
            vars = [m.getvar('wgt'), m.getvar('bias')]
            state_vars = state_vars + vars
        self.state_saver = tf.train.Saver(state_vars)
        self.saved_state_path = self.state_saver.save(session, spath, global_step=step)

    def reopen_current_session(self):
        self.current_session = TFT.copy_session(self.current_session)  # Open a new session with same tensorboard stuff
        self.current_session.run(tf.global_variables_initializer())
        self.restore_session_params()  # Reload old weights and biases to continued from where we last left off

    def restore_session_params(self, path=None, sess=None):
        spath = path if path else self.saved_state_path
        session = sess if sess else self.current_session
        self.state_saver.restore(session, spath)

    def close_current_session(self, view=True):
        self.save_session_params(sess=self.current_session)
        TFT.close_session(self.current_session, view=view)


# A general ann module = a layer of neurons (the output) plus its incoming weights and biases.
class Gannmodule():

    def __init__(self, ann, index, invariable, insize, outsize):
        self.ann = ann
        self.insize = insize  # Number of neurons feeding into this module
        self.outsize = outsize  # Number of neurons in this module
        self.input = invariable  # Either the gann's input variable or the upstream module's output
        self.index = index
        self.name = "Module-" + str(self.index)
        self.build()

    def build(self):
        mona = self.name
        n = self.outsize
        self.weights = tf.Variable(np.random.uniform(-.1, .1, size=(self.insize, n)),
                                   name=mona + '-wgt', trainable=True)  # True = default for trainable anyway
        self.biases = tf.Variable(np.random.uniform(-.1, .1, size=n),
                                  name=mona + '-bias', trainable=True)  # First bias vector
        self.output = tf.nn.relu(tf.matmul(self.input, self.weights) + self.biases, name=mona + '-out')
        self.ann.add_module(self)

    def getvar(self, type):  # type = (in,out,wgt,bias)
        return {'in': self.input, 'out': self.output, 'wgt': self.weights, 'bias': self.biases}[type]

    # spec, a list, can contain one or more of (avg,max,min,hist); type = (in, out, wgt, bias)
    def gen_probe(self, type, spec):
        var = self.getvar(type)
        base = self.name + '_' + type
        with tf.name_scope('probe_'):
            if ('avg' in spec) or ('stdev' in spec):
                avg = tf.reduce_mean(var)
            if 'avg' in spec:
                tf.summary.scalar(base + '/avg/', avg)
            if 'max' in spec:
                tf.summary.scalar(base + '/max/', tf.reduce_max(var))
            if 'min' in spec:
                tf.summary.scalar(base + '/min/', tf.reduce_min(var))
            if 'hist' in spec:
                tf.summary.histogram(base + '/hist/', var)


# *********** CASE MANAGER ********
# This is a simple class for organizing the cases (training, validation and test) for a
# a machine-learning system
# To instantiate this object you need to define a function that generates cases on the form
# Case = [input-vector, target-vector], and then the manager will split the cases into training, val, and test.

class Caseman():

    def __init__(self, cfunc, vfrac=0.1, tfrac=0.1):
        self.casefunc = cfunc
        self.validation_fraction = vfrac
        self.test_fraction = tfrac
        self.training_fraction = 1 - (vfrac + tfrac)
        self.generate_cases()
        self.organize_cases()

    def generate_cases(self):
        self.cases = self.casefunc()  # Run the case generator.  Case = [input-vector, target-vector]

    def organize_cases(self):
        ca = np.array(self.cases)
        np.random.shuffle(ca)  # Randomly shuffle all cases
        separator1 = round(len(self.cases) * self.training_fraction)
        separator2 = separator1 + round(len(self.cases) * self.validation_fraction)
        self.training_cases = ca[0:separator1]
        self.validation_cases = ca[separator1:separator2]
        self.testing_cases = ca[separator2:]

    def get_training_cases(self): return self.training_cases

    def get_validation_cases(self): return self.validation_cases

    def get_testing_cases(self): return self.testing_cases


#   ****  MAIN functions ****

# After running this, open a Tensorboard (Go to localhost:6006 in your Chrome Browser) and check the
# 'scalar', 'distribution' and 'histogram' menu options to view the probed variables.
def autoex(epochs=300, nbits=4, lrate=0.03, showint=100, mbs=None, vfrac=0.1, tfrac=0.1, vint=100, sm=False,
           bestk=None):
    size = 2 ** nbits
    mbs = mbs if mbs else size
    case_generator = (lambda: TFT.gen_all_one_hot_cases(2 ** nbits))
    cman = Caseman(cfunc=case_generator, vfrac=vfrac, tfrac=tfrac)
    ann = Gann(dims=[size, nbits, size], cman=cman, lrate=lrate, showint=showint, mbs=mbs, vint=vint, softmax=sm)
    ann.gen_probe(0, 'wgt', ('hist', 'avg'))  # Plot a histogram and avg of the incoming weights to module 0.
    ann.gen_probe(1, 'out', ('avg', 'max'))  # Plot average and max value of module 1's output vector
    ann.add_grabvar(0, 'wgt')  # Add a grabvar (to be displayed in its own matplotlib window).
    ann.run(epochs, bestk=bestk)
    ann.runmore(epochs * 2, bestk=bestk)
    TFT.fireup_tensorboard('probeview')
    return ann


def countex(epochs=5000, nbits=15, ncases=500, lrate=0.5, showint=500, mbs=20, vfrac=0.1, tfrac=0.1, vint=200, sm=True,
            bestk=1):
    case_generator = (lambda: TFT.gen_vector_count_cases(ncases, nbits))
    cman = Caseman(cfunc=case_generator, vfrac=vfrac, tfrac=tfrac)
    ann = Gann(dims=[nbits, nbits * 3, nbits + 1], cman=cman, lrate=lrate, showint=showint, mbs=mbs, vint=vint,
               softmax=sm)
    ann.run(epochs, bestk=bestk)
    TFT.fireup_tensorboard('probeview')
    return ann
