# PyGeoFetch Follow-Up Post

Last week's post on PyGeoFetch reached way more people than I expected, with ninety two percent of the audience outside my own network, and a lot of you saved it rather than just scrolling past. Thank you.

So here is something real, not a pitch.

A few weeks ago I was fixing InSAR coregistration, which is the step that aligns two satellite radar images to a fraction of a pixel before you can measure ground deformation from them.

My first attempt used a standard geolocation solver based on three equations, and it worked ninety four percent of the time in testing. That was not good enough, because it gets called fifty times per interferogram, so that failure rate compounds fast.

My second attempt tried warm starting the solver with each neighboring pixel's answer, which should have helped. Instead it made things worse.

That failure turned out to be the useful part. It proved the problem was not what I thought it was, and that sent me looking in the right place instead of tuning parameters blindly for another two hours.

The actual fix was to stop solving for ground points at all and pull them straight from the DEM's own coordinates instead, using orbit math that is actually reliable.

The result was 49 out of 49 test points verified, on the exact scenario that used to fail a third of the time.

That fix is now running real Sentinel 1 data over Ghana's Obuasi mining district and measuring actual ground subsidence, not a demo.

The part I am proudest of is not the fix itself. It is that the docs still say plainly where the reliability gaps remain, because a tool that hides its limitations is more dangerous than one that states them.

There are 21 real, runnable notebooks, including this one, that open directly in Colab from the docs.

https://appiahkubis14.github.io/pygeofetch-docs/

If you are working with SAR or InSAR, subsidence monitoring, or you just like reading about bugs that took three tries to actually fix, I would love to hear from you.
