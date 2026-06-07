# Log instancia Z (FOCUS: real-augment + cross-motor)

No solapa con A (threshold-cv), B (loo-validation), C (window-features).


## Real-augment (jitter/escalado de fallos reales) — DESCARTADO
Hipótesis: aumentar las secuencias de fallo REALES (en vez de inyectar sintéticos en normales)
reduciría la varianza por estar ancladas a la verdad.

- v1 (perturbar toda la traza): peor en todo (M1=0.23, M5=0.41 @s42).
- v2 (perturbar SOLO la región de fallo, preservando baseline):
  - realaug @s42: M1=0.50, M2=0.77, M4=0.72, M5=0.42, M6=0.78 — peor que champ3.
  - realboth (synth+real) @s42: M1=0.71, M2=0.28(!), M4=0.69, M5=0.93, M6=0.86 — ERRÁTICO.
- Veredicto: NO reduce varianza (M2 colapsa a 0.28). La inyección sintética sigue siendo mejor.

## Pivote -> cross-motor features
M2 y M4 fallan en la misma sesión. Probar features de contexto inter-motor para esos dos.

## Cross-motor M4->M2 — WIN ROBUSTO ✅
Hipótesis: M2 y M4 co-fallan en la misma sesión; dar a M2 features de contexto de M4
(temperature_diff, roll_std_20, temperature) debería estabilizar M2.

- Asimetría: dar contexto de M4 a M2 AYUDA; dar contexto de M2 a M4 PERJUDICA (M4 ya saturado).
  Por eso solo se aplica a M2.
- M2 en 4 semillas:
  - champ3:  0.883 / 0.905 / 0.905 / 0.292  -> media 0.746
  - crossm2: 0.905 / 0.872 / 0.905 / 0.648  -> media 0.833 (+0.087, y mitiga el colapso s3)
- Veredicto: ACEPTADO. cross_feats {2: [M4 temp_diff, M4 roll_std_20, M4 temperature]}.

## Siguiente: combinar crossm2 + ensemble (best)

## best_cross = CAMPEÓN de la instancia Z ✅
Apila las dos mejoras de robustez: ensemble por motor (M2/M4/M5) + cross-motor (M4->M2).

M2 en 4 semillas (media / mínimo):
- champ3:      0.746 / 0.292   (colapso s3)
- crossm2:     0.833 / 0.648
- best (ens):  0.852 / 0.693
- best_cross:  0.880 / 0.809   <- colapso eliminado, ambas mejoras se acumulan

best_cross medias por motor (4 semillas): M1≈0.53, M2≈0.88, M4≈0.85, M5≈0.82, M6≈0.93.
(M3 sin medir aquí; n_seeds=1, sin cross -> ~igual que champ3.)
Macro medio ≈ 0.80.

Inestables restantes: M1 (~0.53, sin partner cross claro) y M5 (colapsa a 0.49 en s1).
Próximo: buscar contexto cross para M1/M5 o señal de anomalía global inter-motor.

## gctx (contexto de anomalía global para M1/M5) — DESCARTADO
Añadir max inter-motor de roll_std/temp_diff como contexto a M1 y M5.
- M1: 0.526 -> 0.382 (peor). M5: 0.824 -> 0.648 (peor).
- Causa probable: las filas sintéticas llevan el valor global del normal de origen (obsoleto),
  enseñando correlación equivocada. El max global lo domina el motor más ruidoso.
- Veredicto: DESCARTADO. M2-cross se mantiene; M1/M5 siguen sin partner útil.

## RESUMEN FINAL instancia Z (foco real-augment + cross-motor)
- CAMPEÓN: best_cross = ensemble por motor (M2/M4/M5) + cross-motor M4->M2.
  Medias 4 semillas: M1≈0.53, M2≈0.88, M4≈0.85, M5≈0.82, M6≈0.93. Macro≈0.80.
- Contribución clave: ESTABILIZAR M2 (mín 0.292 -> 0.809) combinando ensemble + contexto de M4.
- Descartados: real-augment, contexto global inter-motor.
- Sin resolver: M1 (~0.53, inestable) y los colapsos ocasionales de M5.
