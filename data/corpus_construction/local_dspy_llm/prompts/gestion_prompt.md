You are a structured extraction assistant specializing in Peruvian print journalism.

    The input is OCR text from a scanned page or region of a Peruvian newspaper belonging to the El Comercio editorial group (Trome, Peru21, Ojo, etc.). These are Spanish-language publications covering Peruvian national and international news, politics, economy, culture and sport.

    OCR quality may be imperfect: expect broken words, missing accents, inconsistent spacing and mixed-case artifacts. Use journalistic context to interpret ambiguous text —do not "correct" it in the output, preserve it as-is.

    Given the OCR text below, extract every distinct article or editorial piece found.

    Rules:
- Text is primarily Spanish; preserve it as-is, including OCR artifacts.
- Preserve reading order (top-to-bottom, left-to-right column order).
- A region or page may contain multiple unrelated articles — extract each separately.
- Exclude captions, page numbers, section headers and advertisements.
- Do not invent, infer or complete missing fields.
- Return [] if no articles are present.
- Return ONLY the JSON array. No explanation, preamble, or markdown fences.

Return ONLY a JSON array. Each element must have exactly these keys:
- "title"       : main headline, as it appears (empty string if absent)
- "subheadline" : secondary headline, deck, or standfirst (empty string if absent)
- "author"      : byline — typically introduced by "Por", "Redacción", or a staff name (empty string if absent)
- "body"        : full article body text, preserving paragraph breaks with \n (empty string if absent)

--- EXAMPLES ---
### Example 1
OCR TEXT:
```text
## DIPLOMACIA

## Donald Trump y Vladimir Putin podrían reunirse antes de julio

Asesor de seguridad de Trum se encuentra en una controversia por haberse reunidoconembajador ruso antes de elecciones.

(Reuters)EIKremlin dijo que se ha conversado sobre una posible reunión entre el presidente Vladimir Putin y su homólogo estadounidense, Donakd Trump, antes del encuentrodelG-20enjulio, pero que no hay nada especifico hasta ahora.

Sin embargo, el nuevo Gobierno estadounidense está bajo presión por sus lazos con Rusia, luego de que Michael Flynn, asesor de segu-

Amboslíderesnuncase han reunido, pero han dicho que quierentratarde reparar los maltrechos lazos entre Estados Unidos y Rusia, que cayeron a su nivel más bajo desde la Guerra Fría después de la anexión rusa de la península ucranianadeCrimeaen 2014.

## REPORTE

ds

Petémka. Gobiemo de Trump niega que uno de sus asesores de seguridad harya discutido con Moscu sobre lirvrantamiento de sandiones.

<!-- image -->

## PROCESO JUDICIAL

## Analizarán motivaciones de Trump para veto migratorio

El fiscal generat del estado de Washington prometio descubrir "lo que realmente motivó" el decreto de Trump sobre inmigración, un paso que podría provocar un inusual análisis público sobre cómo un mandatario de EE.UU. toma decisiones de seguridad nacional. El fiscal general de Washington, Bob Ferguson, señaló que buscará

con ahínco obtener los documentos escritos y los correos electrónicos redactados por funcionarios del Gobierno que podrían contener evidencia de que la orden ejecutiva era inconstitucionalmente sesgada contra los musulmanes o el Islam. Expertos legales dicen que esto podría llevar a los tribunales a aguas

inexploradas.

## INVESTIGACIÓN

## México todavía no detecta aumento de deportaciones desde Estados Ünidos

## Fiscalía chilena viaja a Brasil para recopilar información de OAS

(Reuters)ElcancillerdeMéxico, Luis Videgaray, dijo que el número de mexicanos deportadosde EE.UU. no haaumentado en el comienzo del Gobierno de Donald Trump, en medio de reportes de operativoscontrainmigrantesilegales envariasciudadesdeesepaís.

El funcionario dijo a Televisa que el númerodedeportaciones se manteníaestable comparado con el año pasaridad nacional de Trump. busca superar una controversia originada por una conversación telefónica que sos. tuvocon elembajador de Rusia en Estados Unidos, Sergei Kislyak, antes de que Trump asumieraelcargo.

Luts Videgaray.

<!-- image -->

"Obviamente, cada embajadorinforma alcentro (Mos. cú) sobre todos los contactos que tiene para quela informaciónllegueanosotros, perono estamos dispuestos acomentar las discusiones internas que se sost ienenen Washington", dijo Peskov.

## Sanciones

Cuandoselepreguntósobre esto, el portavoz del Kremlin, DmitryPeskov,dijoenunaconferenciatelefónicaqueFlynny Kislyak nohabíanconversado sobre el levantamiento delas sanciones contra Moscú.

Altosfuncionariosdela Casa Blanca hanestadorevisando el fin de semana loscontactos de Flynn para indagar si discutióla posibilidad de levantar las sanciones estadounidenses contra Rusia una vez que Trumpasumieraelcargo, lo cual pod ría violar una ley que prohíbe a ciudd; 10 particular sv lcra Sen políticaext e or

## CoreadelNorte

'S n.capaces de defender contra un ataque de un misil balístico norcoreano y vamosa tomartodaslasmedidas necesarias para disuadír y derrotar amenazas a nuestroterritorioyciudadanos y a los de nuestros aliados", dijo el portavoz del Pentágono.

## BREVES

## VENEZUELA Vicepresidente en lista negra

dn

<!-- image -->

EE.UU. incluyóalvicepresidente de Venezuela, Tareck El Aissami, y a uno de sus colaboradores en una lista negra por narcotráfico, la primera medida del Gobierno de Donald Trumpcontrafuncionarios de la administración del presidente Nicolás Maduro por lavado de dinero y tráficodedrogas.

## BRASIL Niega Intención de protección

El presidente de Brasil, Michel Temer, dijo que nointerfeririaenlasinvestigacionessobre corrupciónen el país y negó haber designado a un estrechoasesor como ministrode su Gabinete para protegerlode un eventual procesamiento judicial.

De otro lado, el Pentágono condenó enérgicamente la másreciente prueba de misil de Corea del Norte y dijo que el compromiso de EF UU.'e protegerelterrito ıp: to a aldo dof: u es"i wu n ravle"

do y que incluso se ubicaba ligeramente por debajo.

No obstante, el secretario de Relaciones Exteriores de México destacó que se han triplicadolasllamadas telefónicasdiariasde mexicanos en consulados o representaciones haciendo preguntas o quejas en tornoa los procedimientos de deportaciones.

(GestiontJorge Abbott, el fiscal Nacional de Chile se reunirá este viernes con el ProcuradorGeneralde Brasil, Rodrigo Janot, con mirasacooperarenunainves. tigación sobre aportesirregulares de la constructora OAS a campañas políticas.

"La cifra de deportados se mantiene (...)No hemos visto todavianingúnincremento", dijoVidegaray.

La semana pasada agentes federales arrestaron a cientos de migrantes indocumentadosenal menoscuatrociudades de EE.UU.

El diario chileno La Terceraindicóquelareuniónse enmarca enla convocatoria del fiscal brasileño a las autoridadesdediversos países que vienen investigando posibles actos de corrupción dentro del caso Lava Jato.

La fiscalía chilena dijo en un comunicado que la reuniónbusca"intercambiarinformación, facilitar el contacto con todos los fiscales del caso Lava Jato y atender las peticiones de colaboración internacional que puedan presentarse".

## OTROSÍ DIGO

Panamá. La iscalía especial anticorrupción de Panamá pidió alnterpol emitir una alerta roja para detener a dos híos del expresidente Ricardo Martineli, señalados en el escándalo de sobomos de Odebrecht. Ricardo γ Luls Enrique Martinelli habría recibido más de 20 millones de euros.

La semana pasada, la revista brasileña Veja aseguró que en las declaraciones de los ejecutivos de OAS fue mencionado un supuesto aporte a la çampaña de la actual presidenta chilena, Michelle Bachelet, quien negó tajantemente la acusación.
```
JSON ARRAY:
```json
[
  {
    "title": "Donald Trump y Vladimir Putin podrían reunirse antes de julio",
    "body": "(Reuters) El Kremlin dijo que se ha conversado sobre una posible reunión entre el presidente Vladimir Putin y su homólogo estadounidense, Donald Trump, antes del encuentro del G-20 en julio, pero que no hay nada específico hasta ahora. Ambos líderes nunca se han reunido, pero han dicho que quieren tratar de reparar los maltrechos lazos entre Estados Unidos y Rusia, que cayeron a su nivel más bajo desde la Guerra Fría después de la anexión rusa de la península ucraniana de Crimea en 2014. Sin embargo, el nuevo Gobierno estadounidense está bajo presión por sus lazos con Rusia, luego de que Michael Flynn, asesor de seguridad nacional de Trump, busca superar una controversia originada por una conversación telefónica que sostuvo con el embajador de Rusia en Estados Unidos, Sergei Kislyak, antes de que Trump asumiera el cargo. Sanciones. Altos funcionarios de la Casa Blanca han estado revisando el fin de semana los contactos de Flynn para indagar si discutió la posibilidad de levantar las sanciones estadounidenses contra Rusia una vez que Trump asumiera el cargo, lo cual podría violar una ley que prohíbe a ciudadanos particulares involucrarse en política exterior. Cuando se le preguntó sobre esto, el portavoz del Kremlin, Dmitry Peskov, dijo en una conferencia telefónica que Flynn y Kislyak no habían conversado sobre el levantamiento de las sanciones contra Moscú. 'Obviamente, cada embajador informa al centro (Moscú) sobre todos los contactos que tiene para que la información llegue a nosotros, pero no estamos dispuestos a comentar las discusiones internas que se sostienen en Washington', dijo Peskov. Corea del Norte. De otro lado, el Pentágono condenó enérgicamente la más reciente prueba de misil de Corea del Norte y dijo que el compromiso de EE.UU. de proteger el territorio patrio y a aliados como Japón y Corea del Sur es 'invulnerable'. 'Somos capaces de defender contra un ataque de un misil balístico norcoreano y vamos a tomar todas las medidas necesarias para disuadir y derrotar amenazas a nuestro territorio y ciudadanos y a los de nuestros aliados', dijo el portavoz del Pentágono."
  },
  {
    "title": "Analizarán motivaciones de Trump para veto migratorio",
    "body": "El fiscal general del estado de Washington prometió descubrir 'lo que realmente motivó' el decreto de Trump sobre inmigración, un paso que podría provocar un inusual análisis público sobre cómo un mandatario de EE.UU. toma decisiones de seguridad nacional. El fiscal general de Washington, Bob Ferguson, señaló que buscará con ahínco obtener los documentos escritos y los correos electrónicos redactados por funcionarios del Gobierno que podrían contener evidencia de que la orden ejecutiva era inconstitucionalemente sesgada contra los musulmanes o el Islam. Expertos legales dicen que esto podría llevar a los tribunales a aguas inexploradas."
  },
  {
    "title": "Vicepresidente en lista negra",
    "body": "EE.UU. incluyó al vicepresidente de Venezuela, Tareck El Aissami, y a uno de sus colaboradores en una lista negra por narcotráfico, la primera medida del Gobierno de Donald Trump contra funcionarios de la administración del presidente Nicolás Maduro por lavado de dinero y tráfico de drogas."
  },
  {
    "title": "Niega intención de protección",
    "body": "El presidente de Brasil, Michel Temer, dijo que no interferiría en las investigaciones sobre corrupción en el país y negó haber designado a un estrecho asesor como ministro de su Gabinete para protegerlo de un eventual procesamiento judicial."
  },
  {
    "title": "México todavía no detecta aumento de deportaciones desde Estados Unidos",
    "body": "(Reuters) El canciller de México, Luis Videgaray, dijo que el número de mexicanos deportados de EE.UU. no ha aumentado en el comienzo del Gobierno de Donald Trump, en medio de reportes de operativos contra inmigrantes ilegales en varias ciudades de ese país. El funcionario dijo a Televisa que el número de deportaciones se mantenía estable comparado con el año pasado y que incluso se ubicaba ligeramente por debajo. 'La cifra de deportados se mantiene (...) No hemos visto todavía ningún incremento', dijo Videgaray. No obstante, el secretario de Relaciones Exteriores de México destacó que se han triplicado las llamadas telefónicas diarias de mexicanos en consulados o representaciones haciendo preguntas o quejas en torno a los procedimientos de deportaciones. La semana pasada agentes federales arrestaron a cientos de migrantes indocumentados en al menos cuatro ciudades de EE.UU."
  },
  {
    "title": "Fiscalía chilena viaja a Brasil para recopilar información de OAS",
    "body": "(Gestión) Jorge Abbott, el fiscal Nacional de Chile se reunirá este viernes con el Procurador General de Brasil, Rodrigo Janot, con miras a cooperar en una investigación sobre aportes irregulares de la constructora OAS a campañas políticas. El diario chileno La Tercera indicó que la reunión se enmarca en la convocatoria del fiscal brasileño a las autoridades de diversos países que vienen investigando posibles actos de corrupción dentro del caso Lava Jato. La fiscalía chilena dijo en un comunicado que la reunión busca 'intercambiar información, facilitar el contacto con todos los fiscales del caso Lava Jato y atender las peticiones de colaboración internacional que puedan presentarse'. La semana pasada, la revista brasileña Veja aseguró que en las declaraciones de los ejecutivos de OAS fue mencionado un supuesto aporte a la campaña de la actual presidenta chilena, Michelle Bachelet, quien negó tajantemente la acusación."
  },
  {
    "title": "Panamá. La fiscalía especial anticorrupción de Panamá pidió a Interpol emitir una alerta roja para detener a dos hijos del expresidente Ricardo Martinelli, señalados en el escándalo de sobornos de Odebrecht. Ricardo y Luis Enrique Martinelli habría recibido más de 20 millones de euros.",
    "body": "Panamá. La fiscalía especial anticorrupción de Panamá pidió a Interpol emitir una alerta roja para detener a dos hijos del expresidente Ricardo Martinelli, señalados en el escándalo de sobornos de Odebrecht. Ricardo y Luis Enrique Martinelli habría recibido más de 20 millones de euros."
  }
]
```

--- END EXAMPLES ---

Now extract from the following OCR text:
OCR TEXT:
{ocr_text}
JSON ARRAY:
