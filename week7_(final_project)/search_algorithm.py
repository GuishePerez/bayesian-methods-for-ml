class FacialComposit:
    MAX_SCORE = 10

    def __init__(self, decoder, latent_size):
        self.latent_size = latent_size
        self.latent_placeholder = tf.placeholder(tf.float32, (1, latent_size))
        self.decode = decoder(self.latent_placeholder)
        self.samples = None
        self.images = None
        self.rating = None
        self.best_image = None
        self.worst_image = None

    def _get_image(self, latent):
        img = sess.run(self.decode,
                       feed_dict={self.latent_placeholder: latent[None, :]})[0]
        img = np.clip(img, 0, 1)
        return img

    def _show_images(self):
        clear_output()
        plt.figure(figsize=(3 * len(self.images), 3))
        ix = np.argsort(self.rating)
        order = 1
        for i in ix:
            plt.subplot(1, len(self.images), order)
            plt.imshow(self.images[i])
            plt.title(str(self.rating[i]))
            plt.axis('off')
            order += 1
        plt.show()

    def show_evolution(self):
        self._show_images()

    @staticmethod
    def _draw_border(image, w=2):
        bordred_image = image.copy()
        bordred_image[:, :w] = [1, 0, 0]
        bordred_image[:, -w:] = [1, 0, 0]
        bordred_image[:w, :] = [1, 0, 0]
        bordred_image[-w:, :] = [1, 0, 0]
        return bordred_image

    @staticmethod
    def _compare_pair(image1, image2):
        clear_output()
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(image1)
        plt.subplot(1, 2, 2)
        plt.imshow(image2)
        plt.show()
        rate = None
        while rate is None:
            try:
                rate = int(input('Select the image that fits the most 0 - left, 1 - right: ').strip())
                assert rate == 0 or rate == 1
            except Exception as e:
                print(e)
                rate = None
        return rate

    def query_initial(self, n_start=5, select_top=None):
        '''
        Creates initial points for Bayesian optimization
        Generate *n_start* random images and asks user to rank them.
        Gives maximum score to the best image and minimum to the worst.
        :param n_start: number of images to rank initialy.
        :param select_top: number of images to keep
        '''
        select_top = n_start
        self.samples = np.random.normal(loc=0.0, scale=1.0, size=(
        select_top, self.latent_size))  ### YOUR CODE HERE (size: select_top x 64 x 64 x 3)
        self.images = np.stack(
            [self._get_image(s) for s in self.samples])  ### YOUR CODE HERE (size: select_top x 64 x 64 x 3)
        self.rating = np.array([0] * select_top)  ### YOUR CODE HERE (size: select_top)

        ### YOUR CODE:
        ### Show user some samples (hint: use self._get_image and input())
        for i, j in combinations(range(select_top), 2):
            rate = self._compare_pair(self.images[i, :, :, :], self.images[j, :, :, :])
            if rate:
                self.rating[j] += 1
            else:
                self.rating[i] += 1
        self.best_image = np.argmax(self.rating)
        self.worst_image = np.argmin(self.rating)

        # Check that tensor sizes are correct
        np.testing.assert_equal(self.rating.shape, [select_top])
        np.testing.assert_equal(self.images.shape, [select_top, 64, 64, 3])
        np.testing.assert_equal(self.samples.shape, [select_top, self.latent_size])

    def process_rate(self, rate, case, ix):
        """
        Decide if we need search further, and if not, what is the candidate_rate
        Args:
          (bool) rate: True if a new image is better than compared
          (string) case: can be best, worst or median
          (int) ix: index of a compared image
        """
        if case == 0:  # best
            if rate:
                # if the current image is better than the best make it the best one
                self.best_image = len(self.images) - 1
                return self.rating[ix] + 1
        elif case == 1:  # worst
            if not rate:
                # if the current image is worst than the worst make it the worst one
                self.worst_image = len(self.images) - 1
                return self.rating[ix] - 1
        elif case == 2:  # median
            if rate:
                return (self.rating[self.best_image] + self.rating[ix]) // 2
            else:
                return (self.rating[self.worst_image] + self.rating[ix]) // 2

        return None

    def evaluate(self, candidate):
        '''
        Queries candidate vs known image set.
        Adds candidate into images pool.
        :param candidate: latent vector of size 1xlatent_size
        '''

        initial_size = len(self.images)
        candidate = candidate[0]
        self.samples = np.concatenate((self.samples, [candidate]), 0)
        image = self._get_image(candidate)
        self.images = np.concatenate((self.images, [image]), 0)

        # compare with the best, worst and median cases at the moment
        to_compare_list = self.best_image, self.worst_image, np.argsort(self.rating)[len(self.rating) // 2]
        candidate_rating = None
        case = 0

        while candidate_rating is None:
            to_compare_image = self.images[to_compare_list[case]]
            rate = self._compare_pair(to_compare_image, image)
            candidate_rating = self.process_rate(rate, case, to_compare_list[case])
            case += 1

        self.rating = np.concatenate((self.rating, [candidate_rating]))

        assert len(self.images) == initial_size + 1
        assert len(self.rating) == initial_size + 1
        assert len(self.samples) == initial_size + 1
        return candidate_rating

    def optimize(self, n_start=5, n_iter=10, w=4, acquisition_type='MPI', acquisition_par=0.3):
        if self.samples is None:
            self.query_initial(n_start=n_start)

        bounds = [{'name': 'z_{0:03d}'.format(i),
                   'type': 'continuous',
                   'domain': (-w, w)}
                  for i in range(self.latent_size)]
        optimizer = GPyOpt.methods.BayesianOptimization(f=self.evaluate, domain=bounds,
                                                        acquisition_type=acquisition_type,
                                                        acquisition_par=acquisition_par,
                                                        exact_eval=False,  # Since we are not sure
                                                        model_type='GP',
                                                        X=self.samples,
                                                        Y=self.rating[:, None],
                                                        maximize=True)
        optimizer.run_optimization(max_iter=n_iter, eps=-1)

    def get_best(self):
        return self.images[self.best_image]

    def draw_best(self, title=''):
        image = self.get_best()
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.show()

The basic hypothesis is it is easier for a human to make a binary choice than set a real number to an arbitrary image.

Thus, on an each iteration we suggest a person to define which image in a pair fits the requirements better. As a result, an image receive a relative estimation, and finally the best image will have the highest score.

On an initialization phase the complete pairwise comparison is performed: all images versus all images.

Then, it would be too consuming to estimate the next image versus all previous. So, we suggest to compare an image with the best one first. If a person considers a new image is better, a new image gets a score higher than the best by one and consequently it becomes a new best image. Otherwise, an image is compared with the worst image. If a person considers a new image is worse, a new image gets a score lower than the worst by one and consequently it becomes a new worst image. If we didn't get a score even now, we compare an image with the median. Based on a decision, we set a score as a half between the median and the best of the worst cases accordingly.


class FacialComposit:
    def __init__(self, decoder, latent_size):
        self.latent_size = latent_size
        self.latent_placeholder = tf.placeholder(tf.float32, (1, latent_size))
        self.decode = decoder(self.latent_placeholder)
        self.samples = None
        self.images = None
        self.rating = None

    def _get_image(self, latent):
        img = sess.run(self.decode,
                       feed_dict={self.latent_placeholder: latent[None, :]})[0]
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def _show_images(images, titles):
        assert len(images) == len(titles)
        clear_output()
        plt.figure(figsize=(3 * len(images), 3))
        n = len(titles)
        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.imshow(images[i])
            plt.title(str(titles[i]))
            plt.axis('off')
        plt.show()

    @staticmethod
    def _draw_border(image, w=2):
        bordred_image = image.copy()
        bordred_image[:, :w] = [1, 0, 0]
        bordred_image[:, -w:] = [1, 0, 0]
        bordred_image[:w, :] = [1, 0, 0]
        bordred_image[-w:, :] = [1, 0, 0]
        return bordred_image

    def query_initial(self, n_start=10, select_top=6):
        '''
        Creates initial points for Bayesian optimization
        Generate *n_start* random images and asks user to rank them.
        Gives maximum score to the best image and minimum to the worst.
        :param n_start: number of images to rank initialy.
        :param select_top: number of images to keep
        '''
        self.samples = np.zeros((select_top, self.latent_size))  ### YOUR CODE HERE (size: select_top x 64 x 64 x 3)
        self.images = np.zeros((select_top, 64, 64, 3))  ### YOUR CODE HERE (size: select_top x 64 x 64 x 3)
        self.rating = np.zeros((select_top))  ### YOUR CODE HERE (size: select_top)

        ### YOUR CODE:
        ### Show user some samples (hint: use self._get_image and input())
        images = []
        codes = []
        score = []

        for i in range(select_top):
            latent_code = np.random.normal(size=self.latent_size) * 2
            img = self._get_image(latent_code)
            images.append(img)
            codes.append(latent_code)

        print("Rank these images based on how similar they are to the person on a scale of 1(low) to 10(high):")
        self._show_images(images, ["Image" + str(i) for i in range(1, len(images) + 1)])
        scores = input("Enter your scores from left to right, separeted by ','").split(',')

        indices = np.argsort(scores)

        for i in range(select_top):
            self.samples[i, :] = codes[indices[i]]
            self.images[i, :, :, :] = images[indices[i]]
            self.rating[i] = scores[indices[i]]

        # Check that tensor sizes are correct
        np.testing.assert_equal(self.rating.shape, [select_top])
        np.testing.assert_equal(self.images.shape, [select_top, 64, 64, 3])
        np.testing.assert_equal(self.samples.shape, [select_top, self.latent_size])

    def evaluate(self, candidate):
        '''
        Queries candidate vs known image set.
        Adds candidate into images pool.
        :param candidate: latent vector of size 1xlatent_size
        '''
        initial_size = len(self.images)

        ### YOUR CODE HERE
        ## Show user an image and ask to assign score to it.
        ## You may want to show some images to user along with their scores
        ## You should also save candidate, corresponding image and rating
        print(candidate.shape)
        rand_ind = np.random.randint(low=0, high=self.images.shape[0], size=5)
        rand_img = [self.images[rand_ind[i]] for i in range(5)]
        rand_score = [self.rating[rand_ind[i]] for i in range(5)]
        print("You have evaluated these pictures as follows:")
        # self._show_images(rand_img,rand_score)
        candidate_image = self._get_image(candidate[0])
        rand_img.append(candidate_image)
        rand_score.append(str("Candidate"))
        self._show_images(rand_img, rand_score)
        candidate_rating = int(input("Enter your score for the candidate image:"))
        self.images = np.vstack((self.images, np.array([candidate_image])))
        self.samples = np.vstack((self.samples, candidate))
        self.rating = np.hstack((self.rating, np.array([candidate_rating])))

        assert len(self.images) == initial_size + 1
        assert len(self.rating) == initial_size + 1
        assert len(self.samples) == initial_size + 1
        return candidate_rating

    def optimize(self, n_iter=10, w=4, acquisition_type='MPI', acquisition_par=0.3):
        if self.samples is None:
            self.query_initial()

        bounds = [{'name': 'z_{0:03d}'.format(i),
                   'type': 'continuous',
                   'domain': (-w, w)}
                  for i in range(self.latent_size)]
        optimizer = GPyOpt.methods.BayesianOptimization(f=self.evaluate, domain=bounds,
                                                        acquisition_type=acquisition_type,
                                                        acquisition_par=acquisition_par,
                                                        exact_eval=False,  # Since we are not sure
                                                        model_type='GP',
                                                        X=self.samples,
                                                        Y=self.rating[:, None],
                                                        maximize=True)
        optimizer.run_optimization(max_iter=n_iter, eps=-1)

    def get_best(self):
        index_best = np.argmax(self.rating)
        return self.images[index_best]

    def draw_best(self, title=''):
        index_best = np.argmax(self.rating)
        image = self.images[index_best]
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.show()





The
algorithm
that
I
have
implemented
has
the
following
features:

Initially
it
displays
some
image and asks
the
user
to
rank
them.This
gives
an
initial
estimate
of
the
values
of
the
function
that
we
want
to
maximize.
The
ranking is done
by
the
user
on
a
scale
of
1 - 10,
with 1 being the lowest resemblence and 10 being the highest resemblence.
Candidate
images
are
generated and the
user is asked
to
rank
those
images in the
same
ranking
system.This
allowes
the
bayesian
optimization
algorithm
to
discover
new
portions
of
the
function.
After
showing
a
number
of
images
the
final
maximum
of
the
function is estimated.
The
reference
images
are
sampled
according
to
the
acquisition
function, MPI.The
drawbacks
of
this
algorithm
are:

This
algorithm
gives
highly
correlated
samples, very
often.
The
ranking
function is determined
by
the
user, and is based
on
the
users
perception.Thus
it is bound
to
have
bias.
Very
often
the
algorithm
gets
stuck in a
local
optimum
leading
to
highly
correlated
samples
around
that
optimum.This
leads
to
the
user
giving
false
data
to
force
the
sampler
to
sample
away
from that optimum