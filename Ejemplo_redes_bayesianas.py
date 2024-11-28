# Ejemplo básico de redes bayesianas con pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Definir la estructura de la red bayesiana
# Nodo A (lluvia) afecta a Nodo B (césped mojado) y Nodo C (conducir peligroso)
red = BayesianNetwork([
    ('Lluvia', 'CespedMojado'),
    ('Lluvia', 'ConducirPeligroso')
])

# Definir las tablas de probabilidad condicional (CPT)
# Probabilidad de que llueva
cpd_lluvia = TabularCPD(
    variable='Lluvia',
    variable_card=2,
    values=[[0.7], [0.3]],  # 70% No llueve, 30% Llueve
    state_names={'Lluvia': ['No', 'Si']}
)

# Probabilidad de césped mojado dado si llueve o no
cpd_cesped = TabularCPD(
    variable='CespedMojado',
    variable_card=2,
    values=[[0.9, 0.2], [0.1, 0.8]],  # Dependencia: (No llueve, Llueve)
    evidence=['Lluvia'],
    evidence_card=[2],
    state_names={
        'CespedMojado': ['No', 'Si'],
        'Lluvia': ['No', 'Si']
    }
)

# Probabilidad de conducir peligroso dado si llueve o no
cpd_conducir = TabularCPD(
    variable='ConducirPeligroso',
    variable_card=2,
    values=[[0.6, 0.3], [0.4, 0.7]],  # Dependencia: (No llueve, Llueve)
    evidence=['Lluvia'],
    evidence_card=[2],
    state_names={
        'ConducirPeligroso': ['No', 'Si'],
        'Lluvia': ['No', 'Si']
    }
)

# Añadir las probabilidades a la red
red.add_cpds(cpd_lluvia, cpd_cesped, cpd_conducir)

# Verificar la validez del modelo
if red.check_model():
    print("La red bayesiana es válida.\n")

# Realizar inferencias
inference = VariableElimination(red)

# Ejemplo 1: Probabilidad de que el césped esté mojado
print("Probabilidad de que el césped esté mojado:")
print(inference.query(variables=['CespedMojado']).values, "\n")

# Ejemplo 2: Probabilidad de que esté lloviendo dado que conducir es peligroso
print("Probabilidad de que esté lloviendo dado que conducir es peligroso:")
print(inference.query(variables=['Lluvia'], evidence={'ConducirPeligroso': 'Si'}).values, "\n")

# Ejemplo 3: Probabilidad conjunta de todos los estados
print("Probabilidad conjunta de todos los estados:")
print(inference.query(variables=['Lluvia', 'CespedMojado', 'ConducirPeligroso']))
