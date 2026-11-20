// The chess pieces (shapes + materials), shared by the board on the home page (landing.html)
// and the king-and-queen close-up (duo.html). Needs three.js loaded first.
function pieceKit() {
  const C = (hex) => new THREE.Color(hex).convertSRGBToLinear();
  const P = (o) => { o.color = C(o.color); return new THREE.MeshPhysicalMaterial(o); };
  // pieces get their own, lower environment light so the key light's shading shows (lit side vs. darker side)
  // (base colour a little below #cfcfcc, because the key light brightens it: the lit side then shows about #cfcfcc on screen)
  const matWhite  = P({ color: 0xa9a9a6, roughness: .35, clearcoat: .3, clearcoatRoughness: .25, envMapIntensity: .3, emissive: 0x000000 });
  const matBlack  = P({ color: 0x0e0f11, roughness: .2, clearcoat: 1, clearcoatRoughness: .06, envMapIntensity: .6, emissive: 0x000000 });
  const slitOnWhite = P({ color: 0x3a3b3f, roughness: .5 });
  const slitOnBlack = P({ color: 0x5a5c61, roughness: .4 });

  // ---------- pieces: smooth hourglass shapes (a side profile spun around = lathe) ----------
  const V = (r, y) => new THREE.Vector2(r, y);
  function lathe(points, segs) {
    const smooth = new THREE.SplineCurve(points).getPoints(segs || 80);
    const geo = new THREE.LatheGeometry(smooth, 56); geo.computeVertexNormals(); return geo;
  }
  const hourglass = (h, foot, waist, collar) => lathe([
    V(0, 0), V(foot, 0), V(foot, .03), V(foot * .86, .07), V(waist * 1.25, h * .32), V(waist, h * .55),
    V(collar * .85, h * .9), V(collar, h - .015), V(collar * .9, h), V(0, h + .005)]);
  const sphere = (r, y) => new THREE.SphereGeometry(r, 36, 22).translate(0, y, 0);

  const geos = {};
  geos.p = [[hourglass(.4, .3, .085, .17), sphere(.13, .5)], []];                       // pawn: hourglass + ball
  {                                                                                      // rook: notched top
    const parts = [hourglass(.48, .33, .11, .22), new THREE.CylinderGeometry(.21, .22, .2, 48).translate(0, .58, 0)];
    for (let i = 0; i < 4; i++) {
      const a = i * Math.PI / 2 + Math.PI / 4;
      parts.push(new THREE.BoxGeometry(.11, .08, .11).rotateY(-a).translate(.145 * Math.cos(a), .72, .145 * Math.sin(a)));
    }
    geos.r = [parts, []];
  }
  geos.b = [[hourglass(.56, .32, .09, .17),                                              // bishop: egg top + slit
             lathe([V(0, .53), V(.11, .56), V(.15, .66), V(.13, .78), V(.06, .87), V(0, .885)], 40), sphere(.04, .92)],
            [new THREE.BoxGeometry(.025, .2, .3).rotateX(.62).translate(0, .73, 0)]];
  {                                                                                      // knight: one curved arc
    const R = .17, tube = .085, A = Math.PI * .95;
    const arc = new THREE.TorusGeometry(R, tube, 24, 48, A).translate(-R, .5, 0);
    const endX = -R + R * Math.cos(A), endY = .5 + R * Math.sin(A);
    geos.n = [[hourglass(.5, .33, .1, .17), arc, sphere(tube, .5), sphere(tube, endY).translate(endX, 0, 0)], []];
  }
  geos.q = [[hourglass(.78, .35, .1, .19),                                               // queen: crown ring + ball
             lathe([V(0, .77), V(.15, .79), V(.11, .86), V(.07, .9), V(0, .91)], 30),
             new THREE.TorusGeometry(.105, .022, 16, 48).rotateX(Math.PI / 2).translate(0, .93, 0), sphere(.06, 1.0)], []];
  geos.k = [[hourglass(.88, .36, .105, .2),                                              // king: cap + small cross
             lathe([V(0, .87), V(.16, .89), V(.13, .97), V(0, .99)], 30),
             new THREE.BoxGeometry(.05, .22, .05).translate(0, 1.09, 0),
             new THREE.BoxGeometry(.16, .05, .05).translate(0, 1.11, 0)], []];

  return { matWhite, matBlack, slitOnWhite, slitOnBlack, geos };
}
