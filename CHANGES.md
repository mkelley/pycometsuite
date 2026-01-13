# Change log

## v1.2.0 (2026-01-13)

### API changes

* `cometsuite.instruments.Camera`: To support the bug fix below, all parameters
  are assumed to use the FITS WCS convention: x, y.  ``shape`` and ``centeryx``
  has been removed, replaced with ``size`` and ``crpix``.  Parameters ``scale``,
  and ``center`` have been renamed ``cdelt``, and ``crval``, respectively.
  Input parameters are no longer stored as attributes on the camera.  Instead,
  access the relevant parameters through the ``wcs`` attribute.

### Bug fixes

* `cometsuite.instruments.Camera`:
  - Fixed confusion between x, y and y, x in array shape and center.
  - Aligned data with the center of the pixels.
