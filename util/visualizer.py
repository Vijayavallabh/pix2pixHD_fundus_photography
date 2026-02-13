import numpy as np
import os
import ntpath
import time
import csv
from . import util
from . import html
import scipy.misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.tf_log = opt.tf_log
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.metric_keys = ['G_GAN', 'G_GAN_Feat', 'G_VGG', 'G_SSIM', 'G_GradVar', 'D_real', 'D_fake']
        if self.tf_log:
            import tensorflow as tf
            self.tf = tf
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            self.writer = tf.summary.FileWriter(self.log_dir)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

        self.metrics_dir = os.path.join(opt.checkpoints_dir, opt.name, 'metrics')
        util.mkdirs(self.metrics_dir)
        self.metrics_csv_name = os.path.join(self.metrics_dir, 'loss_history.csv')
        if not os.path.exists(self.metrics_csv_name) or os.path.getsize(self.metrics_csv_name) == 0:
            with open(self.metrics_csv_name, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    'epoch', 'iters', 'time',
                    'G_GAN', 'G_GAN_Feat', 'G_VGG', 'G_SSIM', 'G_GradVar', 'D_real', 'D_fake',
                    'G_total', 'D_mean', 'D_gap'
                ])

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):
        if self.tf_log: # show images in tensorboard output
            img_summaries = []
            for label, image_numpy in visuals.items():
                # Write the image to a string
                try:
                    s = StringIO()
                except:
                    s = BytesIO()
                scipy.misc.toimage(image_numpy).save(s, format="jpeg")
                # Create an Image object
                img_sum = self.tf.Summary.Image(encoded_image_string=s.getvalue(), height=image_numpy.shape[0], width=image_numpy.shape[1])
                # Create a Summary value
                img_summaries.append(self.tf.Summary.Value(tag=label, image=img_sum))

            # Create and write Summary
            summary = self.tf.Summary(value=img_summaries)
            self.writer.add_summary(summary, step)

        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%d.jpg' % (epoch, label, i))
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.jpg' % (epoch, label))
                    util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=30)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_%s_%d.jpg' % (n, label, i)
                            ims.append(img_path)
                            txts.append(label+str(i))
                            links.append(img_path)
                    else:
                        img_path = 'epoch%.3d_%s.jpg' % (n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims)/2.0))
                    webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

        def _to_float(value):
            if isinstance(value, (int, float, np.floating)):
                return float(value)
            return 0.0

        g_gan = _to_float(errors.get('G_GAN', 0.0))
        g_gan_feat = _to_float(errors.get('G_GAN_Feat', 0.0))
        g_vgg = _to_float(errors.get('G_VGG', 0.0))
        g_ssim = _to_float(errors.get('G_SSIM', 0.0))
        g_gradvar = _to_float(errors.get('G_GradVar', 0.0))
        d_real = _to_float(errors.get('D_real', 0.0))
        d_fake = _to_float(errors.get('D_fake', 0.0))

        g_total = (
            g_gan
            + g_gan_feat
            + g_vgg
            + self.opt.lambda_ssim * g_ssim
            + self.opt.lambda_gradvar * g_gradvar
        )
        d_mean = 0.5 * (d_real + d_fake)
        d_gap = d_real - d_fake

        with open(self.metrics_csv_name, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                epoch, i, t,
                g_gan, g_gan_feat, g_vgg, g_ssim, g_gradvar, d_real, d_fake,
                g_total, d_mean, d_gap
            ])

    def save_training_plots(self):
        if not os.path.exists(self.metrics_csv_name):
            print('No metrics CSV found, skipping training plots.')
            return

        with open(self.metrics_csv_name, 'r', newline='') as csv_file:
            rows = list(csv.DictReader(csv_file))

        if not rows:
            print('No metric rows found, skipping training plots.')
            return

        def read_series(key):
            values = []
            for row in rows:
                try:
                    values.append(float(row.get(key, 0.0)))
                except (TypeError, ValueError):
                    values.append(0.0)
            return np.array(values, dtype=np.float32)

        x = np.arange(1, len(rows) + 1, dtype=np.int32)

        g_total = read_series('G_total')
        g_gan = read_series('G_GAN')
        g_gan_feat = read_series('G_GAN_Feat')
        g_vgg = read_series('G_VGG')
        g_ssim = read_series('G_SSIM')
        g_gradvar = read_series('G_GradVar')
        d_real = read_series('D_real')
        d_fake = read_series('D_fake')
        d_mean = read_series('D_mean')
        d_gap = read_series('D_gap')

        fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

        axes[0].plot(x, g_total, label='G_total', linewidth=2)
        axes[0].plot(x, g_gan, label='G_GAN', alpha=0.8)
        axes[0].plot(x, g_gan_feat, label='G_GAN_Feat', alpha=0.8)
        axes[0].plot(x, g_vgg, label='G_VGG', alpha=0.8)
        axes[0].plot(x, g_ssim, label='G_SSIM', alpha=0.8)
        axes[0].plot(x, g_gradvar, label='G_GradVar', alpha=0.8)
        axes[0].set_title('Generator Losses')
        axes[0].set_ylabel('Loss')
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc='best')

        axes[1].plot(x, d_real, label='D_real', linewidth=1.8)
        axes[1].plot(x, d_fake, label='D_fake', linewidth=1.8)
        axes[1].plot(x, d_mean, label='D_mean', linewidth=1.5, linestyle='--')
        axes[1].set_title('Discriminator Losses')
        axes[1].set_xlabel('Log Step')
        axes[1].set_ylabel('Loss')
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc='best')

        fig.tight_layout()
        combined_plot_path = os.path.join(self.metrics_dir, 'loss_curves.png')
        fig.savefig(combined_plot_path, dpi=180)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(x, d_gap, label='D_gap (D_real - D_fake)', linewidth=1.8)
        ax2.axhline(0.0, color='black', linewidth=1, linestyle='--')
        ax2.set_title('Discriminator Balance Monitor')
        ax2.set_xlabel('Log Step')
        ax2.set_ylabel('Gap')
        ax2.grid(alpha=0.25)
        ax2.legend(loc='best')
        fig2.tight_layout()
        gap_plot_path = os.path.join(self.metrics_dir, 'd_balance.png')
        fig2.savefig(gap_plot_path, dpi=180)
        plt.close(fig2)

        print('Saved training metrics to %s' % self.metrics_csv_name)
        print('Saved loss plots to %s and %s' % (combined_plot_path, gap_plot_path))

    def save_training_summary(self):
        if not os.path.exists(self.metrics_csv_name):
            print('No metrics CSV found, skipping summary generation.')
            return

        with open(self.metrics_csv_name, 'r', newline='') as csv_file:
            rows = list(csv.DictReader(csv_file))

        if not rows:
            print('No metric rows found, skipping summary generation.')
            return

        def _float(row, key, default=0.0):
            try:
                return float(row.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        best_idx = None
        best_score = None

        for idx, row in enumerate(rows):
            g_total = _float(row, 'G_total')
            d_gap = abs(_float(row, 'D_gap'))
            d_mean = _float(row, 'D_mean')
            score = g_total + 0.2 * d_gap + 0.05 * d_mean

            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx

        start_idx = max(0, len(rows) - 20)
        recent_rows = rows[start_idx:]
        recent_g_total = [_float(r, 'G_total') for r in recent_rows]
        recent_d_gap_abs = [abs(_float(r, 'D_gap')) for r in recent_rows]

        recent_g_mean = float(np.mean(recent_g_total)) if recent_g_total else 0.0
        recent_g_std = float(np.std(recent_g_total)) if recent_g_total else 0.0
        recent_gap_mean = float(np.mean(recent_d_gap_abs)) if recent_d_gap_abs else 0.0

        best_row = rows[best_idx]
        best_epoch = int(float(best_row.get('epoch', 0)))
        best_iters = int(float(best_row.get('iters', 0)))

        summary_lines = [
            'Training Metrics Summary',
            '=========================',
            'Total logged steps: %d' % len(rows),
            '',
            'Heuristic best log step',
            '-----------------------',
            'epoch: %d' % best_epoch,
            'iters: %d' % best_iters,
            'score: %.6f (lower is better)' % best_score,
            'G_total: %.6f' % _float(best_row, 'G_total'),
            'G_GAN: %.6f' % _float(best_row, 'G_GAN'),
            'G_GAN_Feat: %.6f' % _float(best_row, 'G_GAN_Feat'),
            'G_VGG: %.6f' % _float(best_row, 'G_VGG'),
            'G_SSIM: %.6f' % _float(best_row, 'G_SSIM'),
            'G_GradVar: %.6f' % _float(best_row, 'G_GradVar'),
            'D_real: %.6f' % _float(best_row, 'D_real'),
            'D_fake: %.6f' % _float(best_row, 'D_fake'),
            'D_gap: %.6f' % _float(best_row, 'D_gap'),
            '',
            'Recent-window monitor (last up to 20 log points)',
            '-----------------------------------------------',
            'mean G_total: %.6f' % recent_g_mean,
            'std  G_total: %.6f' % recent_g_std,
            'mean |D_gap|: %.6f' % recent_gap_mean,
            '',
            'Early-stop suggestion',
            '---------------------',
            'If holdout visuals are flat/worse and recent mean G_total does not improve',
            'across 3 checkpoint saves, stop and keep the best earlier checkpoint.'
        ]

        summary_path = os.path.join(self.metrics_dir, 'training_summary.txt')
        with open(summary_path, 'w') as summary_file:
            summary_file.write('\n'.join(summary_lines) + '\n')

        best_json_path = os.path.join(self.metrics_dir, 'best_step.json')
        with open(best_json_path, 'w') as json_file:
            json_file.write('{\n')
            json_file.write('  "best_index": %d,\n' % best_idx)
            json_file.write('  "epoch": %d,\n' % best_epoch)
            json_file.write('  "iters": %d,\n' % best_iters)
            json_file.write('  "score": %.8f\n' % best_score)
            json_file.write('}\n')

        print('Saved training summary to %s' % summary_path)
        print('Saved best-step metadata to %s' % best_json_path)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.jpg' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
