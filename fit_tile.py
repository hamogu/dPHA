import logging
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from astropy import table
from sherpa.astro.data import DataPHA
from sherpa.models.basic import Gauss1D, PowLaw1D
from sherpa.stats import Cash
from sherpa.optmethods import LevMar, NelderMead
from sherpa.estmethods import Confidence
from sherpa.fit import Fit

from sherpa.models.model import BinaryOpModel


def flatten_sherpa_model(m):
    if type(m) is BinaryOpModel:
        return flatten_sherpa_model(m.lhs) + flatten_sherpa_model(m.rhs)
    else:
        return [m]


def single_gauss(data, conf):
    '''Initialize sherpa model of Gauss plus powerlaw

    This routine creates a model instance, sets initial values,
    and sets up parameters contraints (e.g. minimum amplitude of
    the Gaussian) to reduce the risk of running the fit into
    unphysical situations.

    Parameter
    ---------
    data : `sherpa.data.Data1D`
        Date to be fit. This is used to guess the initial values
        of the model.
    conf : `dict` or `astropy.table.Row`
        Initial values for FWHM, position, and powerlaw slope

    Returns
    -------
    model : sherpa model
    comp : list
        List of the individual components in the Sherpa model
    '''
    g = Gauss1D()
    p = PowLaw1D()

    x = data.get_indep(filter=True)[0]
    y = data.get_dep(filter=True)
    # Set up starting values that will be close to final results.
    # That is important for numerical stability, and also for speed.
    g.pos = conf['gpos']
    g.pos.max = conf['gpos'] + 100
    g.pos.min = conf['gpos'] - 100
    g.ampl = y[np.argmin(np.abs(x - g.pos.val))]
    #g.fwhm.max = 100
    #g.fwhm.min = 30
    g.fwhm = conf['fwhm']
    g.fwhm.frozen = True
    g.ampl.min = 0

    p.ref = x[0]
    p.gamma = conf['pgamma']
    p.ampl = y[0]

    return p + g, [p, g]


def si_gauss(data, conf):
    '''Like ``sinlge_gauss`` except with 2 Gaussians'''
    model, comp = single_gauss(data, conf)
    g = Gauss1D(name='Si2')
    g.fwhm = comp[1].fwhm
    g.pos = comp[1].pos + 40
    # do not want to link parameters, just set starting value
    g.ampl = comp[1].ampl.val / 5
    return model + g, [comp[0], comp[1], g]


def extract_tile(evt, row):
    '''From an event list, extract just events in a single tile.

    Parameters
    ----------
    evt : `astropy.table.Table`
        Event list
    row : `astropy.table.Row` or dict
        must have elements xlo, xhi, ylo, yhi

    Returns
    -------
    tile : `astropy.table.Table`
        Filtered event list
    '''
    tile = evt[(evt['chipx'] >= row['xlo']) &
               (evt['chipx'] <= row['xhi']) &
               (evt['chipy'] >= row['ylo']) &
               (evt['chipy'] <= row['yhi'])]
    return tile


def fit_tile(d, fitstartrow, init_model, stat, method, errors=False,
             min_counts=50):
    '''Run fit for one tile for one element

    Parameters
    ----------
    d : `sherpa.astro.data.DataPHA`
        Data for one tile
    fitstartrow : `dict` or `atropy.table.Row`
        Specification of start values for fit (element, pos, fwhm etc.)
    init_model : `dict`
        Dictionary that has, for each element, a function that makes
        the initial model
    stat : `sherpa.stats.Stat`
        Sherpa statistic object, e.g. Chi2 or Cash
    method : `sherpa.optmethods.OptMethod`
        Sherpa optimizer, e.g. LevMar
    errors : bool
        If ``True`` estimate uncertainties for each fit
    min_counts : int
        Only perform fits if the data has at least ``min_counts`` counts

    Returns
    -------
    gfit : `sherpa.fit.Fit` or ``None``
        Fit result object with reference to best-fit model values. If
        no fit was run, ``None`` is returned.
    fit_err : `sherpa.fit.ErrorEstResults`
        Error results object that contains the estimated errors on the
        gaussian position. If not error estimation was run, ``None``.
    '''
    d.notice(None, None)
    d.notice(fitstartrow['low'], fitstartrow['hi'])
    if d.get_dep(filter=True).sum() >= min_counts:
        model, comp = init_model[fitstartrow['element']](d, fitstartrow)
        gfit = Fit(d, model, stat=stat, method=method)
        gfit.fit()
        gfit.estmethod = Confidence()
        if errors:
            return gfit, gfit.est_errors(parlist=[comp[1].pos])
        else:
            return gfit, None
    else:
        return None, None


def fit_allelem(d, fitstart, *args, **kwargs):
    '''Run fits for one tile for all elements

    This routine loops over all elements. For each element,
    it selects the right PHA range, bins that data and calls
    ``fit_tile``.

    Parameters
    ----------
    d : `sherpa.astro.data.DataPHA`
        Data for one tile
    fitstart : `astropy.table.Table`
        Table with stargin values for each element
    args, kwargs :
        All other parameters are passed to ``fit_tile``

    Returns
    -------
    fit_res : list
        List of sherpa fit result objects
    fit_err : list
        List of sherpa confidence objects
    '''
    fit_res = []
    fit_err = []
    for i, conf in enumerate(fitstart):
        gfit, err_est = fit_tile(d, conf, *args, **kwargs)
        fit_res.append(gfit)
        fit_err.append(err_est)
    return fit_res, fit_err


def empty_loc_from_template(elements):
    '''Initialize an empty table in the `loc` format the Nick uses

    This routine reads in one of Nick's files and takes the definition
    of the tiles (x_lo, x_high, etc.) from that file. That way, we
    know that the tiling in the new file will be the same.

    Parameters
    ----------
    elements : list
        List of elements for the column ins the loc table

    Returns
    -------
    loc : `astropy.table.Table`
    '''
    loc = table.Table.read('ecs_pha_fits/e1/i3_noTG.lloc', format='ascii',
                           header_start=1)
    loc.keep_columns(['xlo', 'xhi', 'ylo', 'yhi'])
    for e in elements:
        loc[e] = np.nan
        loc[e + 'err'] = np.nan
        loc[e + 'err_up'] = np.nan
        loc[e + 'err_lo'] = np.nan
    return loc


def fit_all(evt, fitstart, *args, **kwargs):
    loc = empty_loc_from_template(fitstart['element'])
    for i, row in enumerate(loc):
        tile = extract_tile(evt, row)
        for conf in fitstart:
            hist, edges = np.histogram(tile['pha'],
                                       bins=np.arange(-0.5,
                                                      np.max(tile['pha'])))
            d = DataPHA('reg', np.arange(np.max(tile['pha'])), hist)
            gfit, err_est = fit_tile(d, conf, *args, **kwargs)
            if gfit is not None:
                # Unfortunately, the Fit object just has list of all parameters,
                # so we need to find the right one by name
                for p in gfit.model.pars:
                    if p.fullname == 'gauss1d.pos':
                        row[conf['element']] = p.val
            # We only run confidence for one parameter in fit_tile, so no need to
            # loop through the list to find the right one
            if err_est is not None:
                row[conf['element'] + 'err_up'] = err_est.parmaxes[0]
                row[conf['element'] + 'err_lo'] = err_est.parmins[0]
                # If the error is unconstrained (too few counts), the output is None
                if (err_est.parmaxes[0] is not None) and (err_est.parmins[0] is not None):
                    row[conf['element'] + 'err'] = max(err_est.parmaxes[0], -err_est.parmins[0])
                else:
                    row[conf['element'] + 'err'] = np.inf
    return loc


def reg2im1024(tab, col):
    '''Reformat entries in the column of an lloc table into a (1024, 1024) array


    Parameters
    ----------
    tab : `astropy.table.Table`
        lloc table
    col : string
        Column name

    Returns
    -------
    im : `numpy.ndarray`
        (1024, 1024) image
    '''
    # could do the same with np.reshape and np.repeat, since it's a regular grid
    # but this is easy enough.
    im = np.zeros((1024, 1024))
    for row in tab:
        im[row['xlo'] - 1 : row['xhi'], row['ylo'] - 1 : row['yhi']] = row[col]
    return im


def plot_allelem(d, fitstart, fit_res, fit_err=None):
    '''Plot a grid with fits for Si, S, Ar, Ca, and Fe for a single tile

    Parameters
    ----------
    d : `sherpa.astro.data.DataPHA`
        Data for one tile
    fitstart : `astropy.table.Table`
        Starting values for fit (for binning parameters for histrogram
        and element labels)
    fit_res : list of Sherpa fit results
        for the model to be plotted
    fit_err : list of Sherpa confidences results or ``None``
        If not ``None``, then 1 sigma confidence intervals for the
        position will be marked.
    '''
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    for i, conf in enumerate(fitstart):
        ax = axes.flatten()[i]
        d.notice(None, None)
        d.notice(conf['low'], conf['hi'])
        x, y, yerr, xerr, xlabel, ylabel = d.to_plot()
        ax.plot(x, y, 'ko', label='Data')
        ylim = ax.get_ylim()
        if fit_res[i] is not None:
            # data could be binned, but we want model in small bins
            x = np.arange(conf['low'], conf['hi'], .1)
            ax.plot(x, fit_res[i].model(x), linewidth=2, label='model')
            for comp in flatten_sherpa_model(fit_res[i].model):
                ax.plot(x, comp(x), linewidth=2, label='model')
        ax.set_title(conf['element'])
        ax.set_ylim([0, ylim[1]])
        ax.set_xlim(conf['low'], conf['hi'])
        if fit_err is not None:
            err = fit_err[i]
            if err is not None:
                low_bound = conf['low'] if err.parmins[0] is None else \
                    err.parvals[0] + err.parmins[0]
                hi_bound = conf['hi'] if err.parmaxes[0] is None else \
                    err.parvals[0] + err.parmins[0]
                ax.axvspan(low_bound, hi_bound, alpha=.5)
    for ax in axes[1, :]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    return fig, axes


def plot_loc_table(loc, evts, elements,
                   subplotskw={'nrows': 2, 'ncols': 3, 'figsize': (12, 8),
                               'subplot_kw': {'aspect': 'equal'},
                               'gridspec_kw': {'wspace': .5}}):
    '''Plot a grid of PHA images  for Si, S, Ar, Ca, and Fe

    Parameters
    ----------
    loc : `astropy.table.Table`
        lloc table
    evts : `astropy.table.Table`
        Event table (for display of the data)
    elements : list of strings
        Element names to be plotted from the lloc table
    subplotskw : dict
        arguments passed on to `plt.subplots`
    '''

    fig, axes = plt.subplots(**subplotskw)
    for i, elem in enumerate(elements):
        ax = axes.flatten()[i]
        im = ax.imshow(reg2im1024(loc, elem).T, origin='lower')
        ax.set_title(elem)
        out = plt.colorbar(im, ax=ax)
        out.set_label('PHA [channel]')

    total = np.histogram2d(evts['chipx'], evts['chipy'],
                           bins=[np.arange(0, 1025, 32),
                                 np.arange(0, 1025, 128)])
    im = axes[1, 2].imshow(total[0].T, origin='lower',
                           extent=[0, 1024, 0, 1024],
                           cmap=plt.get_cmap('inferno'),
                           vmin=0)
    cbar = plt.colorbar(im, ax=axes[1, 2])
    cbar.set_label('counts / bin')
    axes[1, 2].set_title('counts / bin')
    return fig, axes


def fit_obsid_list(obsids):
    '''Fit one or more obsids and write standard output products.

    This looks for evt list called `data/12345/repro/evt1_notgain_filts.fits`,
    where 12345 is the ObsID. If more than one ObsID is given, data will be
    merged. The routine write lloc files and summary pdf plots.

    Default setting for fitter and optimizer on the module level are used
    for the fits.

    Parameters
    ----------
    obsids : list
        One or more obsids. Data from multiple obsids will be merged before
        fitting.
    '''
    logger.info('Working on ObsIDs: ' + ','. join(str(obsids)))
    evts = [table.Table.read(os.path.join('data', str(obsid),
                                          'repro/evt1_notgain_filts.fits'),
                             hdu='EVENTS')
            for obsid in obsids]
    evt = table.vstack(evts, metadata_conflicts='silent')
    for i, ccd in enumerate(ccdid):
        logger.info(f'Working on CCD {ccd}')
        evtccd = evt[evt['ccd_id'] == i]
        if len(evtccd) > 1000:
            loc = fit_all(evtccd, fitstart, init_model,
                          errors=True, stat=stat, method=opt)
            path = os.path.join('casa_pha_fits', 'o'.join([str(o) for o in obsids]))
            os.makedirs(path, exist_ok=True)
            loc.write(os.path.join(path, f'{ccd}_noTG.lloc'), format='ascii')
            fig, axes = plot_loc_table(loc, evtccd, fitstart['element'])
            fig.savefig(os.path.join(path, f'{ccd}_imges.pdf'))


# Sherpa generates quite some output on screen,
# We'll silence some of that with these lines
#sherpablog = logging.getLogger('sherpa')
#sherpablog.setLevel(logging.WARNING)

# stat = Chi2Gehrels()
stat = Cash()
opt = NelderMead()

init_model = defaultdict(lambda: single_gauss, Si=si_gauss)
ccdid = ['i0', 'i1', 'i2', 'i3', 's0', 's1', 's2', 's3', 's4', 's5']
logger = logging.getLogger(__name__)
logger.setLevel('INFO')
fitstart = table.Table.read('fit_start.txt', format='ascii')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fit position of lines in PHA space')
    parser.add_argument('obsid', action='append', help='ObsID to process (several obsids can be given, in which case event lists will be merged before fitting).')
    args = parser.parse_args()
    fit_obsid_list(args.obsid)
